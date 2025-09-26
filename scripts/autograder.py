"""
Autograder Management Script
Automates interaction with the autograder API with nice formatting and automatic log retrieval.
"""

import requests
import json
import time
import sys
import os
from typing import Dict, Any, Optional
from datetime import datetime
import argparse
from datetime import datetime

GROUP_NUMBER = 27
BASE_URL = "http://dl-berlin.ecn.purdue.edu/api"
GH_TOKEN = os.environ.get("GH_TOKEN", "")


class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_header(text: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}\n")


def print_success(text: str):
    print(f"{Colors.OKGREEN}[+] {text}{Colors.ENDC}")


def print_error(text: str):
    print(f"{Colors.FAIL}[!] {text}{Colors.ENDC}")


def print_info(text: str):
    print(f"{Colors.OKCYAN}[>] {text}{Colors.ENDC}")


def print_warning(text: str):
    print(f"{Colors.WARNING}[?] {text}{Colors.ENDC}")


def make_request(
    endpoint: str, method: str = "POST", data: Optional[Dict] = None
) -> Optional[Dict]:
    url = f"{BASE_URL}/{endpoint}"

    if data is None:
        data = {}

    data["group"] = GROUP_NUMBER
    data["gh_token"] = GH_TOKEN

    try:
        if method == "GET":
            response = requests.get(url, json=data, timeout=30)
        else:
            response = requests.post(url, json=data, timeout=30)

        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print_error(f"Request failed: {e}")
        return None


def print_test_results(data: Dict[str, Any], indent: int = 0):
    prefix = "  " * indent

    for key, value in data.items():
        if key in [
            "autograder_run_log",
            "system_run_log",
            "start_time",
            "end_time",
            "run_time",
        ]:
            continue

        if isinstance(value, dict):
            if key != "Total":
                print(f"{prefix}{Colors.BOLD}{key}:{Colors.ENDC}")
                print_test_results(value, indent + 1)
        else:
            if key == "Total":
                print(f"{prefix}{Colors.BOLD}Total: {Colors.ENDC}{value}")
            elif isinstance(value, (int, float)):
                status = Colors.OKGREEN if value > 0 else Colors.FAIL
                symbol = "[+]" if value > 0 else "[!]"
                print(f"{prefix}{status}{symbol} {key}: {value}{Colors.ENDC}")
            else:
                print(f"{prefix}{key}: {value}")


def schedule_run():
    print_header("Scheduling Autograder Run")
    result = make_request("schedule")

    if result:
        print_success("Run scheduled successfully!")
        if isinstance(result, dict):
            for key, value in result.items():
                print_info(f"{key}: {value}")
    else:
        print_error("Failed to schedule run")

    return result


def monitor_runs(wait: bool = True, poll_interval: int = 10):
    """Monitor all runs and optionally wait for completion."""
    print_header("Monitoring Autograder Runs")

    if not wait:
        result = make_request("run/all", "GET")
        if result:
            print(json.dumps(result, indent=2))
        return result

    print_info(
        f"Polling every {poll_interval} seconds for Group {GROUP_NUMBER}... (Press Ctrl+C to stop)\n"
    )

    try:
        iteration = 0
        while True:
            iteration += 1
            result = make_request("run/all", "GET")

            if not result and not isinstance(result, list):
                print_error("Failed to fetch run status")
                time.sleep(poll_interval)
                continue

            result_str = json.dumps(result)
            group_found = str(GROUP_NUMBER) in result_str

            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] Check #{iteration}: ", end="")

            if isinstance(result, list):
                if not result:
                    print_success("No active runs")
                    print_success(f"Group {GROUP_NUMBER} run completed!")
                    break

                active_runs = []
                for i, value in enumerate(result):
                    status = value if isinstance(value, str) else str(value)
                    active_runs.append(f"{i}={status}")

                print(f"{len(result)} active run(s): {result}")

                if not group_found:
                    print_success(
                        f"\nGroup {GROUP_NUMBER} no longer in queue - run completed!"
                    )
                    break
                else:
                    print_info(f"Group {GROUP_NUMBER} still running...")
            else:
                print(f"Unexpected response: {result}")

            # Wait before next check
            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print_warning("\n\nMonitoring stopped by user")

    return result


def get_best_run():
    print_header("Fetching Best Run Results")
    result = make_request("best_run", "GET")

    if result and isinstance(result, dict):
        if "start_time" in result:
            print_info(f"Start Time: {result['start_time']}")
        if "end_time" in result:
            print_info(f"End Time: {result['end_time']}")
        if "run_time" in result:
            print_info(f"Run Time: {result['run_time']}")

        if "Total" in result:
            print(
                f"\n{Colors.BOLD}{Colors.OKGREEN}Overall Score: {result['Total']}{Colors.ENDC}\n"
            )

        print_test_results(result)

        return result
    else:
        print_error("Failed to fetch best run results")
        return None


def download_log(log_path: str, output_file: Optional[str] = None):
    print_header(f"Downloading Log: {log_path}")

    result = make_request("log/download", "GET", {"log": log_path})

    if result:
        if output_file is None:
            output_file = os.path.basename(log_path)

        with open(output_file, "w") as f:
            if isinstance(result, str):
                f.write(result)
            else:
                f.write(json.dumps(result, indent=2))

        print_success(f"Log saved to: {output_file}")
        return output_file
    else:
        print_error("Failed to download log")
        return None


def main():
    if not GH_TOKEN:
        print_error("GH_TOKEN environment variable not set!")
        print_info("Please set it with: export GH_TOKEN='your_token_here'")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Autograder Management Tool")
    parser.add_argument("--schedule", action="store_true", help="Schedule a new run")
    parser.add_argument(
        "--monitor", action="store_true", help="Monitor runs (one-time check)"
    )
    parser.add_argument("--best", action="store_true", help="Get best run results")
    parser.add_argument("--logs", action="store_true", help="Download best run logs")
    parser.add_argument("--auto", action="store_true", help="Full automated workflow")

    args = parser.parse_args()

    if args.schedule:
        schedule_run()
        return

    if args.monitor:
        monitor_runs(wait=False)
        return

    if args.best:
        get_best_run()
        return

    if args.logs:
        best_run = make_request("best_run", "GET")
        if best_run and "autograder_run_log" in best_run:
            download_log(best_run["autograder_run_log"])
            if "system_run_log" in best_run:
                download_log(best_run["system_run_log"])
        return

    if args.auto:
        schedule_run()
        print_info("\nWaiting 5 seconds before monitoring...")
        time.sleep(5)
        monitor_runs(wait=True)
        best_run = get_best_run()

        return

    print_header("Autograder Management Tool")
    print_info(f"Group: {GROUP_NUMBER}")
    print_info(f"Base URL: {BASE_URL}\n")

    while True:
        print(f"\n{Colors.BOLD}Available Actions:{Colors.ENDC}")
        print("1. Schedule new run")
        print("2. Monitor runs (wait for completion)")
        print("3. Check best run")
        print("4. Download best run logs")
        print("5. Full workflow (schedule + monitor + fetch best + download logs)")
        print("6. Exit")

        choice = input(
            f"\n{Colors.OKCYAN}Select an option (1-6): {Colors.ENDC}"
        ).strip()

        if choice == "1":
            schedule_run()
        elif choice == "2":
            monitor_runs(wait=True)
        elif choice == "3":
            get_best_run()
        elif choice == "4":
            best_run = make_request("best_run", "GET")
            if best_run and "autograder_run_log" in best_run:
                download_log(best_run["autograder_run_log"])
                if "system_run_log" in best_run:
                    download_log(best_run["system_run_log"])
        elif choice == "5":
            # Full workflow
            schedule_run()
            print_info("\nWaiting 5 seconds before monitoring...")
            time.sleep(5)
            monitor_runs(wait=True)
            best_run = get_best_run()
        elif choice == "6":
            print_success("Goodbye!")
            break
        else:
            print_warning("Invalid option, please try again")


if __name__ == "__main__":
    main()
