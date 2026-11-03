import argparse
import yaml
from src.inference_pipeline import run_inference

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="ml", choices=["ml", "dl"])
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    run_inference(
        test_path=args.test_path,
        output_path=args.output_path,
        model_type=args.model_type,
        config_path=args.config,
    )
