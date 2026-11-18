
from src.config import PipelineConfig, logger
from src.data_loader import fetch_data
from src.model_engine import prepare_labeled_splits, run_model_training
from src.visualizer import generate_plots
from src.compiler import compile_dashboard


def main() -> None:
    """Orchestrates and triggers our complete, production-grade pipeline execution."""
    logger.info("Initializing Cosmic Curator Pipeline execution...")
    config = PipelineConfig()
    
    try:
        # Step 1: Query API
        raw_data = fetch_data(config)
        
        # Step 2: Split Labeled Sets
        labeled, targets, train_idx, test_idx = prepare_labeled_splits(raw_data, config)
        
        # Step 3: Train and Evaluate Models
        metrics = run_model_training(labeled, targets, train_idx, test_idx, config)
        
        # Step 4: Render Plotly Charts
        plots = generate_plots(raw_data, metrics, config)
        
        # Step 5: Write Output Page
        output_file = compile_dashboard(raw_data, metrics, plots, config)
        
        print("\n" + "="*50)
        print("COSMIC CURATOR PIPELINE SUCCESS")
        print("="*50)
        print(f"Total Records Extracted:     {len(raw_data):,}")
        print(f"Leaky Model ROC-AUC:        {metrics['leaky_auc']:.3f} (CRITICAL WARNING)")
        print(f"Audited Forest ROC-AUC:     {metrics['forest_auc']:.3f} (Scientific Baseline)")
        print(f"Logistic Regression AUC:    {metrics['logistic_auc']:.3f}")
        print(f"Compiled Page Generated:    {output_file}")
        print("="*50 + "\n")
        
    except Exception as e:
        logger.exception(f"Pipeline execution halted due to unexpected system crash: {e}")


if __name__ == "__main__":
    main()
