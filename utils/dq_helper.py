import os
import sys
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

class DQPipelineManager:
    """
    Manager class to bridge project configurations with Great Expectations and Jupyter Notebooks.
    """
    def __init__(
        self,
        config_path: str = "config/data_quality_config.yaml",
        rules_path: str = "config/validation_rules.yaml"
    ):
        self.root_dir = Path(__file__).resolve().parent.parent
        self.config_path = self.root_dir / config_path
        self.rules_path = self.root_dir / rules_path
        self.config = self._load_yaml(self.config_path)
        self.rules = self._load_yaml(self.rules_path)
        self._setup_logging()

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Loads YAML configuration file."""
        if not path.exists():
            print(f"Warning: Configuration file not found at {path}")
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _setup_logging(self):
        """Initializes simple logging format."""
        logging.basicConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        self.logger = logging.getLogger("DQPipelineManager")

    def get_stage_config(self, stage_name: str) -> Dict[str, Any]:
        """Returns configuration for a specific ETL stage (e.g., ingestion, staging, curated)."""
        stages = self.config.get("stages", {})
        if stage_name not in stages:
            raise KeyError(f"Stage '{stage_name}' not found in configuration.")
        return stages[stage_name]

    def get_table_rules(self, table_name: str) -> Dict[str, Any]:
        """Returns table validation rules defined in validation_rules.yaml."""
        tables = self.rules.get("tables", {})
        return tables.get(table_name, {})

    def load_dataset(self, asset_name: Optional[str] = None, stage_name: str = "ingestion") -> pd.DataFrame:
        """
        Loads dataset configured under the specified ETL stage.
        Supports parquet and csv formats.
        """
        stage_cfg = self.get_stage_config(stage_name)
        assets = stage_cfg.get("data_assets", [])
        if not assets:
            raise ValueError(f"No data assets configured for stage '{stage_name}'.")

        target_asset = None
        if asset_name:
            for asset in assets:
                if asset.get("name") == asset_name:
                    target_asset = asset
                    break
            if not target_asset:
                raise ValueError(f"Asset '{asset_name}' not found in stage '{stage_name}'.")
        else:
            target_asset = assets[0]

        file_path = self.root_dir / target_asset["path"]
        fmt = target_asset.get("format", "parquet").lower()

        self.logger.info(f"Loading dataset '{target_asset.get('name')}' from {file_path}")

        if fmt == "parquet":
            return pd.read_parquet(file_path)
        elif fmt == "csv":
            return pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {fmt}")

    def apply_table_rules(self, gx_dataset, table_name: str) -> int:
        """
        Dynamically applies expectation rules from validation_rules.yaml to a Great Expectations dataset object.
        """
        rules = self.get_table_rules(table_name)
        columns = rules.get("columns", {})
        applied_count = 0

        for col, col_config in columns.items():
            expectations = col_config.get("expectations", [])
            for exp in expectations:
                exp_type = exp.get("type")
                kwargs = exp.get("kwargs", {}).copy()

                # If lookup_file is provided, dynamically populate value_set from lookup file
                if "lookup_file" in kwargs:
                    lookup_rel_path = kwargs.pop("lookup_file")
                    lookup_col = kwargs.pop("lookup_column", "LocationID")
                    lookup_path = self.root_dir / lookup_rel_path
                    if lookup_path.exists():
                        lookup_df = pd.read_csv(lookup_path)
                        kwargs["value_set"] = sorted(lookup_df[lookup_col].dropna().unique().tolist())
                    else:
                        self.logger.warning(f"Lookup file '{lookup_path}' not found.")

                if hasattr(gx_dataset, exp_type):
                    func = getattr(gx_dataset, exp_type)
                    func(col, **kwargs)
                    applied_count += 1
                else:
                    self.logger.warning(f"Expectation '{exp_type}' not supported on Great Expectations dataset.")

        self.logger.info(f"Applied {applied_count} expectation rules for table '{table_name}'.")
        return applied_count

    def evaluate_validation_results(self, validation_result: dict) -> bool:
        """
        Evaluates GX validation results against pipeline quality thresholds.
        Returns True if passed, False if failed.
        """
        success = validation_result.get("success", False)
        statistics = validation_result.get("statistics", {})
        
        success_pct = statistics.get('success_percent')
        success_pct_str = f"{success_pct:.2f}%" if success_pct is not None else "N/A"

        self.logger.info(
            f"Validation run completed: success={success}, "
            f"evaluated_expectations={statistics.get('evaluated_expectations')}, "
            f"success_percent={success_pct_str}"
        )

        critical_action = self.config.get("thresholds", {}).get("critical_failure_action", "abort_pipeline")
        
        if not success and critical_action == "abort_pipeline":
            self.logger.error("Data Quality check FAILED! Threshold action is set to abort pipeline.")
            return False
            
        return True

if __name__ == "__main__":
    manager = DQPipelineManager()
    print("ETL DQ Pipeline Manager successfully loaded configuration:")
    print(f"- Pipeline Name: {manager.config.get('pipeline', {}).get('name')}")
    print(f"- Environment: {os.getenv('ENVIRONMENT', 'development')}")
    df_test = manager.load_dataset()
    print(f"- Default Ingestion Dataset Loaded: Shape {df_test.shape}")

