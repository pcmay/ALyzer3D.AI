import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

// Interface for the structured JSON output from your Python script
interface PythonPredictionResult {
  prediction_label: 'amyloid' | 'non_amyloid';
  prediction_probability: number;
  sequence: string;
  error?: string | null; // Can be a string or null
}

// Interface for the final data stored in Supabase
interface AnalysisResult {
  amyloidogenicity_score: number;
  confidence: number;
  risk_level: string;
  regions_analyzed: number;
  analysis_time_ms: number;
}


Deno.serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const supabase = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_ANON_KEY') ?? '',
      { global: { headers: { Authorization: req.headers.get('Authorization')! } } }
    );

    // --- POST Request: Start a new analysis ---
    if (req.method === 'POST') {
      const formData = await req.formData();
      const sessionId = formData.get('sessionId') as string;
      const jsonFile = formData.get('jsonFile') as File;
      const pdbFile = formData.get('pdbFile') as File;

      if (!sessionId || !jsonFile || !pdbFile) {
        return new Response(
          JSON.stringify({ error: 'Missing required fields: sessionId, jsonFile, pdbFile' }),
          { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        );
      }

      console.log(`Received request for session: ${sessionId}`);

      // Create initial analysis record in Supabase
      const { data: analysisRecord, error: insertError } = await supabase
        .from('analysis_results')
        .insert({
          session_id: sessionId,
          status: 'processing'
        })
        .select()
        .single();

      if (insertError) {
        console.error('Error creating analysis record:', insertError);
        return new Response(JSON.stringify({ error: 'Failed to create analysis record' }), { status: 500, headers: corsHeaders });
      }

      console.log(`Created analysis record with ID: ${analysisRecord.id}`);

      // Upload files to Supabase Storage
      const jsonFileName = `${sessionId}/${analysisRecord.id}_${jsonFile.name}`;
      const pdbFileName = `${sessionId}/${analysisRecord.id}_${pdbFile.name}`;
      
      const [jsonUpload, pdbUpload] = await Promise.all([
        supabase.storage.from('analysis-files').upload(jsonFileName, jsonFile, { upsert: true }),
        supabase.storage.from('analysis-files').upload(pdbFileName, pdbFile, { upsert: true })
      ]);

      if (jsonUpload.error || pdbUpload.error) {
        console.error('File upload errors:', { json: jsonUpload.error, pdb: pdbUpload.error });
        await supabase.from('analysis_results').update({ status: 'failed', error_message: 'File upload failed' }).eq('id', analysisRecord.id);
        return new Response(JSON.stringify({ error: 'File upload failed' }), { status: 500, headers: corsHeaders });
      }

      // Update record with file paths
      await supabase.from('analysis_results').update({
        json_file_path: jsonUpload.data?.path,
        pdb_file_path: pdbUpload.data?.path
      }).eq('id', analysisRecord.id);
      
      // Start the actual analysis in the background without blocking the request
      Deno.serveHttp(req, async () => {
          performAnalysis(supabase, analysisRecord.id, jsonFile, pdbFile);
      });

      return new Response(
        JSON.stringify({ analysisId: analysisRecord.id, message: 'Analysis started successfully' }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // --- GET Request: Poll for analysis results ---
    if (req.method === 'GET') {
      const url = new URL(req.url);
      const analysisId = url.searchParams.get('analysisId');

      if (!analysisId) {
        return new Response(JSON.stringify({ error: 'Missing analysisId parameter' }), { status: 400, headers: corsHeaders });
      }

      const { data: result, error } = await supabase.from('analysis_results').select('*').eq('id', analysisId).single();

      if (error || !result) {
        return new Response(JSON.stringify({ error: 'Analysis not found' }), { status: 404, headers: corsHeaders });
      }

      return new Response(JSON.stringify(result), { headers: { ...corsHeaders, 'Content-Type': 'application/json' } });
    }

    return new Response(JSON.stringify({ error: 'Method not allowed' }), { status: 405, headers: corsHeaders });

  } catch (error) {
    console.error('General error in Deno function:', error);
    return new Response(JSON.stringify({ error: 'Internal server error' }), { status: 500, headers: corsHeaders });
  }
});


/**
 * Performs the core analysis by executing the external Python script.
 * This function handles file I/O, process execution, and cleanup.
 */
async function performAnalysis(
  supabase: any,
  analysisId: string,
  jsonFile: File,
  pdbFile: File
): Promise<void> {
  const startTime = Date.now();
  // Create a unique temporary directory for this specific analysis run
  const tempDir = await Deno.makeTempDir({ prefix: "amyloid_analysis_" });

  try {
    console.log(`[${analysisId}] Starting background analysis in temporary directory: ${tempDir}`);

    const tempJsonPath = `${tempDir}/${jsonFile.name}`;
    const tempPdbPath = `${tempDir}/${pdbFile.name}`;

    // Write the uploaded files to the temporary directory on the server's file system
    await Deno.writeFile(tempJsonPath, new Uint8Array(await jsonFile.arrayBuffer()));
    await Deno.writeFile(tempPdbPath, new Uint8Array(await pdbFile.arrayBuffer()));
    console.log(`[${analysisId}] Temporary files created successfully.`);

    // --- Execute the Python Prediction Script ---
    const command = new Deno.Command("python", { // Use "python3" if that's the command on your system
      args: [
        "./run_prediction.py", // The path to your Python wrapper script
        "--pdb",
        tempPdbPath,
        "--json",
        tempJsonPath,
      ],
    });

    console.log(`[${analysisId}] Executing command: python ${command.args.join(' ')}`);
    const { code, stdout, stderr } = await command.output();

    // --- Handle Python Script Errors ---
    if (code !== 0) {
      const errorOutput = new TextDecoder().decode(stderr);
      console.error(`[${analysisId}] Python script failed with code ${code}:`, errorOutput);
      throw new Error(`Python script execution failed: ${errorOutput}`);
    }

    // --- Process Successful Python Script Output ---
    const resultText = new TextDecoder().decode(stdout);
    const result: PythonPredictionResult = JSON.parse(resultText);

    if (result.error) {
        throw new Error(`Prediction logic failed: ${result.error}`);
    }
    
    console.log(`[${analysisId}] Python script executed successfully. Prediction: ${result.prediction_label}`);
    const analysis_time_ms = Date.now() - startTime;

    // --- Update Supabase with the final, real results ---
    await updateAnalysisResult(supabase, analysisId, result, analysis_time_ms);
    console.log(`[${analysisId}] Analysis completed and results saved to Supabase.`);

  } catch (error) {
    console.error(`[${analysisId}] Analysis process failed:`, error);
    // If anything goes wrong, update the record with a failed status
    await supabase
      .from('analysis_results')
      .update({
        status: 'failed',
        error_message: error.message || 'Unknown analysis failure'
      })
      .eq('id', analysisId);
  } finally {
      // --- Cleanup: CRITICAL to remove temporary files ---
      await Deno.remove(tempDir, { recursive: true });
      console.log(`[${analysisId}] Cleaned up temporary directory: ${tempDir}`);
  }
}

/**
 * Updates the Supabase record with the final results from the analysis.
 */
async function updateAnalysisResult(
  supabase: any,
  analysisId: string,
  result: PythonPredictionResult,
  analysis_time_ms: number
): Promise<void> {

  let risk_level: string;
  if (result.prediction_probability < 0.4) {
    risk_level = 'Low';
  } else if (result.prediction_probability < 0.7) {
    risk_level = 'Medium';
  } else {
    risk_level = 'High';
  }

  const { error } = await supabase
    .from('analysis_results')
    .update({
      status: 'completed',
      // Map the Python output to your Supabase schema
      amyloidogenicity_score: result.prediction_probability,
      confidence: result.prediction_probability, // Using probability as confidence
      risk_level: risk_level,
      regions_analyzed: result.sequence.length, // Using sequence length as regions analyzed
      analysis_time_ms: analysis_time_ms,
    })
    .eq('id', analysisId);

  if (error) {
    // This error will be caught by the calling function's catch block
    throw new Error(`Failed to update Supabase record: ${error.message}`);
  }
}