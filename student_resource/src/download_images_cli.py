import argparse
import os
import pandas as pd

# Reuse the thread-based downloader with retries from utils.py
from utils import download_images

def main():
    parser = argparse.ArgumentParser(description="Download images from a CSV column of URLs.")
    parser.add_argument("--csv", type=str, help="Path to CSV with image URLs", required=False)
    parser.add_argument("--column", type=str, default="image_link", help="Column name for URLs in the CSV")
    parser.add_argument("--out", type=str, required=True, help="Output folder to save images")
    parser.add_argument("--fail-log", type=str, help="Path to a failed_downloads.txt file to retry only failures", required=False)
    parser.add_argument("--limit", type=int, help="Limit number of images (for quick tests)", required=False)
    parser.add_argument("--workers", type=int, help="Max parallel downloads (lower to reduce timeouts)", required=False)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.fail_log:
        if not os.path.exists(args.fail_log):
            print(f"Fail log not found: {args.fail_log}")
            return
        with open(args.fail_log, "r", encoding="utf-8") as f:
            links = [line.strip() for line in f if line.strip()]
        print(f"Retrying {len(links)} failed links from {args.fail_log}...")
    else:
        if not args.csv:
            print("Please provide --csv when not using --fail-log")
            return
        df = pd.read_csv(args.csv)
        if args.column not in df.columns:
            print(f"Column '{args.column}' not found in {args.csv}. Available: {list(df.columns)}")
            return
        links = df[args.column].dropna().astype(str).tolist()
        print(f"Found {len(links)} links in {args.csv} column '{args.column}'.")

    if args.limit:
        links = links[:args.limit]
        print(f"Downloading first {args.limit} links for a quick test.")

    download_images(links, args.out, max_workers=args.workers)
    print("Done.")

if __name__ == "__main__":
    main()