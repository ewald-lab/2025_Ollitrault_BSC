nextflow.enable.dsl=2

// To RUN BSC: nextflow run main.nf -params-file configs/config_base_bsc.json -profile local -resume
// To RUN Axiom: nextflow run main.nf -params-file workflows/config/config_base_axiom.json -resume

include { preprocessing_workflow } from './workflows/01_preprocessing.nf'
include { processing_workflow } from './workflows/02_processing.nf'
include { postprocessing_workflow } from './workflows/03_postprocessing.nf'
//include { evaluation_workflow } from './workflows/04_evaluation.nf'
include { concresponse_workflow } from './workflows/06_concresponse.nf'
include { concresponse_workflow_feature_wise } from './workflows/06_concresponse.nf'
include { concresponse_workflow_bayesian_inference } from './workflows/06_concresponse.nf'
include { analysis_workflow } from './workflows/05_analysis.nf'
include { analysis_workflow_processed } from './workflows/05_analysis.nf'
include { classification_workflow } from './workflows/07_classification.nf'

// Generate a unique run code with date + short UUID
def today = new Date()
def formattedDate = today.format('yyyyMMdd')   // e.g., 20260204
params.run_code = "run_${formattedDate}_" + UUID.randomUUID().toString()[0..7] 


workflow {
    input_path = Channel.fromPath(params.input_path)
    preprocessed_data = preprocessing_workflow(input_path)
    processed_data = processing_workflow(preprocessed_data.path_data_level2_log)
     
    if( params.post_processing ) {
        postprocessed_data = postprocessing_workflow(
            processed_data.path_data_level2,
            processed_data.path_data_level3,
            processed_data.path_data_level4,
            processed_data.path_data_level5,
            processed_data.path_data_level6,
            processed_data.path_data_level5_all,
            processed_data.path_data_level6_all
        )
        analysis_workflow_processed(processed_data.path_data_level2 , processed_data.path_data_level3, processed_data.path_data_level4, processed_data.path_data_level5)
    } else {
        postprocessed_data = processed_data
    }
    
    // evaluation_results = evaluation_workflow(postprocessed_data.path_data_level4, postprocessed_data.path_data_level5)
    concresponse_results = concresponse_workflow(postprocessed_data.path_data_level4, postprocessed_data.path_data_level5)
    // concresponse_results = concresponse_workflow_feature_wise(postprocessed_data.path_data_level4, postprocessed_data.path_data_level5)
    // concresponse_results = concresponse_workflow_bayesian_inference(postprocessed_data.path_data_level4, postprocessed_data.path_data_level5)
    
    if (params.input_name == "axiom") {
        compute_meta_pods = concresponse_results
            .flatten()
            .map { file ->
                def label = file.getName()
                    .replace("pods_axiom_", "")
                    .replace("_original_conc.parquet", "")
                // Only keep the three PODs you care about
                if (!["Metadata_mtt_normalized","Metadata_ldh_normalized","Metadata_Count_Cells"].contains(label)) {
                    return null
                }
                // <-- use tuple() instead of list []
                tuple(label, file)
            }
            .filter { it != null }
        classification_workflow(postprocessed_data.path_data_level4, postprocessed_data.path_data_level6, compute_meta_pods)
    }

    analysis_workflow(postprocessed_data.path_data_level2 , postprocessed_data.path_data_level3, postprocessed_data.path_data_level4, postprocessed_data.path_data_level5)
}
