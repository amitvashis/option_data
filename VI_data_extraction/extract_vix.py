"""
India VIX Historical Data Extraction from NSE
==============================================
Uses Selenium with a VISIBLE browser to navigate the NSE VIX historical page,
interact with the date pickers, and download the CSV data.

NSE aggressively blocks all API calls — this approach works because it uses
the actual page UI exactly like a human would.

Date Range: 2024-08-01 → 2026-02-12
"""

import os
import sys
import time
import glob
import shutil
import pandas as pd
from datetime import date, timedelta

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

try:
    from webdriver_manager.chrome import ChromeDriverManager
except ImportError:
    ChromeDriverManager = None

# ── Configuration ───────────────────────────────────────────────────────────
START_DATE = date(2026, 1, 1)
END_DATE = date(2026, 2, 12)
CHUNK_DAYS = 90  # Safe chunk size for NSE date range limit

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "india_vix_historical.csv")
DOWNLOAD_DIR = os.path.join(OUTPUT_DIR, "downloads_temp")

BASE_URL = "https://www.nseindia.com"
VIX_PAGE = f"{BASE_URL}/reports-indices-historical-vix"

MONTH_MAP = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}

MONTH_FULL = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}


# ── Helper Functions ────────────────────────────────────────────────────────

def date_to_str(d):
    """Convert date to DD-Mon-YYYY (e.g. 01-Aug-2024)."""
    return d.strftime("%d-%b-%Y")


def generate_date_chunks(start, end, chunk_days=CHUNK_DAYS):
    """Split a date range into chunks."""
    chunks = []
    current_start = start
    while current_start <= end:
        current_end = min(current_start + timedelta(days=chunk_days - 1), end)
        chunks.append((current_start, current_end))
        current_start = current_end + timedelta(days=1)
    return chunks


def create_browser():
    """Create a visible Chrome/Edge browser with download directory set."""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    # Try Chrome first
    try:
        options = ChromeOptions()
        # Do NOT use headless — NSE detects it
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-extensions")
        options.add_argument("--window-size=1300,900")
        options.add_argument("--no-sandbox")
        options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
        options.add_experimental_option("useAutomationExtension", False)
        options.add_experimental_option("prefs", {
            "download.default_directory": DOWNLOAD_DIR,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        })

        if ChromeDriverManager:
            service = ChromeService(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
        else:
            driver = webdriver.Chrome(options=options)

        # Mask webdriver property
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
        )
        print("✅ Chrome browser launched.")
        return driver

    except Exception as chrome_err:
        print(f"⚠️  Chrome failed: {chrome_err}")
        print("   Trying Microsoft Edge...")

        from selenium.webdriver.edge.service import Service as EdgeService
        from selenium.webdriver.edge.options import Options as EdgeOptions
        try:
            from webdriver_manager.microsoft import EdgeChromiumDriverManager
        except ImportError:
            EdgeChromiumDriverManager = None

        edge_options = EdgeOptions()
        edge_options.add_argument("--disable-blink-features=AutomationControlled")
        edge_options.add_argument("--window-size=1300,900")
        edge_options.add_argument("--no-sandbox")
        edge_options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
        edge_options.add_experimental_option("prefs", {
            "download.default_directory": DOWNLOAD_DIR,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
        })

        if EdgeChromiumDriverManager:
            service = EdgeService(EdgeChromiumDriverManager().install())
            driver = webdriver.Edge(service=service, options=edge_options)
        else:
            driver = webdriver.Edge(options=edge_options)

        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
        )
        print("✅ Edge browser launched.")
        return driver


def wait_for_page_load(driver, timeout=20):
    """Wait for the NSE page to fully load."""
    try:
        WebDriverWait(driver, timeout).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )
    except Exception:
        pass
    time.sleep(2)


def set_date_input(driver, input_element, target_date):
    """
    Set a date in the NSE date input (Angular Material datepicker or custom input).
    Clears the input and types the date in DD-Mon-YYYY format.
    """
    date_str = target_date.strftime("%d-%m-%Y")

    try:
        # Try to clear and type directly
        input_element.click()
        time.sleep(0.5)
        input_element.send_keys(Keys.CONTROL + "a")
        time.sleep(0.2)
        input_element.send_keys(date_str)
        time.sleep(0.5)
        input_element.send_keys(Keys.ESCAPE)
        time.sleep(0.3)
    except Exception as e:
        # Fallback: use JavaScript to set the value
        driver.execute_script(
            "arguments[0].value = arguments[1]; "
            "arguments[0].dispatchEvent(new Event('input', {bubbles: true})); "
            "arguments[0].dispatchEvent(new Event('change', {bubbles: true}));",
            input_element, date_str
        )
        time.sleep(0.5)


def wait_for_download(timeout=30):
    """Wait for a CSV file to appear in the download directory."""
    start = time.time()
    while time.time() - start < timeout:
        csv_files = glob.glob(os.path.join(DOWNLOAD_DIR, "*.csv"))
        # Filter out partially downloaded files (.crdownload, .tmp)
        partial = glob.glob(os.path.join(DOWNLOAD_DIR, "*.crdownload"))
        partial += glob.glob(os.path.join(DOWNLOAD_DIR, "*.tmp"))
        if csv_files and not partial:
            return csv_files[0]
        time.sleep(1)
    return None


def clear_download_dir():
    """Clear the temporary download directory."""
    if os.path.exists(DOWNLOAD_DIR):
        for f in os.listdir(DOWNLOAD_DIR):
            fp = os.path.join(DOWNLOAD_DIR, f)
            try:
                os.remove(fp)
            except Exception:
                pass


def try_download_via_ui(driver, from_date, to_date):
    """
    Navigate to VIX page, set dates, click submit, then download CSV.
    Returns the path to the downloaded CSV, or None.
    """
    # Navigate to VIX page
    driver.get(VIX_PAGE)
    wait_for_page_load(driver, 20)
    time.sleep(3)

    # Find date inputs — NSE uses id="from" and id="to" OR a custom datepicker
    from_input = None
    to_input = None

    # Try various selectors
    selectors_from = [
        "input#from",
        "input#fromDate",
        "input[name='fromDate']",
        "input[placeholder*='From']",
        "#cr_historical_vix_702 input:first-of-type",
        "#historicalvixDate input:first-of-type",
        "input.datepicker-input-from",
    ]
    selectors_to = [
        "input#to",
        "input#toDate",
        "input[name='toDate']",
        "input[placeholder*='To']",
        "#cr_historical_vix_702 input:last-of-type",
        "#historicalvixDate input:last-of-type",
        "input.datepicker-input-to",
    ]

    # Also try generic approach: find all date inputs
    all_inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='text'], input[type='date'], input.ng-pristine")

    for sel in selectors_from:
        try:
            from_input = driver.find_element(By.CSS_SELECTOR, sel)
            if from_input.is_displayed():
                break
            from_input = None
        except Exception:
            continue

    for sel in selectors_to:
        try:
            to_input = driver.find_element(By.CSS_SELECTOR, sel)
            if to_input.is_displayed():
                break
            to_input = None
        except Exception:
            continue

    # If specific selectors didn't work, try finding date inputs generically
    if not from_input or not to_input:
        # Look for the VIX section by text or container
        date_inputs = driver.find_elements(By.CSS_SELECTOR, "input.hasDatepicker, input[id*='date'], input[id*='Date']")
        if len(date_inputs) >= 2:
            from_input = date_inputs[0]
            to_input = date_inputs[1]
        else:
            # Find all visible text inputs
            visible_inputs = [inp for inp in all_inputs if inp.is_displayed()]
            if len(visible_inputs) >= 2:
                from_input = visible_inputs[0]
                to_input = visible_inputs[1]

    if not from_input or not to_input:
        # Dump page source for debugging
        page_src_snippet = driver.page_source[:3000]
        print(f"   ❌ Could not find date inputs on the page.")
        print(f"   Page title: {driver.title}")
        # Try script-based approach as last resort
        return try_download_via_script(driver, from_date, to_date)

    # Set the from date
    print(f"   Setting From date: {from_date}", end="... ")
    set_date_input(driver, from_input, from_date)

    # Set the to date
    print(f"To date: {to_date}", end="... ")
    set_date_input(driver, to_input, to_date)
    time.sleep(1)

    # Click submit button
    submit_selectors = [
        "button.btn-primary",
        "button#submit",
        "input[type='submit']",
        "button:contains('Submit')",
        "a.btn-primary",
        "button.search-btn",
    ]

    submit_btn = None
    for sel in submit_selectors:
        try:
            submit_btn = driver.find_element(By.CSS_SELECTOR, sel)
            if submit_btn.is_displayed():
                break
            submit_btn = None
        except Exception:
            continue

    if not submit_btn:
        # Try finding by text
        buttons = driver.find_elements(By.TAG_NAME, "button")
        for btn in buttons:
            if any(txt in btn.text.lower() for txt in ["submit", "get data", "search", "filter"]):
                submit_btn = btn
                break

    if submit_btn:
        submit_btn.click()
        print("Submitted.", end=" ")
        time.sleep(5)  # Wait for data to load
    else:
        print("⚠️  No submit button found, trying anyway...", end=" ")

    # Click CSV download button
    clear_download_dir()

    download_selectors = [
        "a[href*='csv']",
        "a[href*='download']",
        "a.download-data-link",
        "#download-csv",
        "button[title*='download']",
        "a[title*='Download']",
        "span.download-data-link",
        "#downloadLink",
    ]

    download_btn = None
    for sel in download_selectors:
        try:
            download_btn = driver.find_element(By.CSS_SELECTOR, sel)
            if download_btn.is_displayed():
                break
            download_btn = None
        except Exception:
            continue

    if not download_btn:
        # Try partial link text
        try:
            download_btn = driver.find_element(By.PARTIAL_LINK_TEXT, "Download")
        except Exception:
            pass

    if not download_btn:
        links = driver.find_elements(By.TAG_NAME, "a")
        for link in links:
            if any(txt in link.text.lower() for txt in ["csv", "download", "export"]):
                download_btn = link
                break

    if download_btn:
        download_btn.click()
        print("Download clicked.", end=" ")

        csv_path = wait_for_download(20)
        if csv_path:
            print(f"✅ Downloaded!")
            return csv_path
        else:
            print("⚠️  Download didn't complete.")
            return None
    else:
        print("⚠️  No download button found. Trying table scrape...")
        return try_scrape_table(driver)


def try_download_via_script(driver, from_date, to_date):
    """Fallback: Use JavaScript to trigger the API call from within the browser."""
    from_str = from_date.strftime("%d-%m-%Y")
    to_str = to_date.strftime("%d-%m-%Y")
    url = f"https://www.nseindia.com/api/historical/vixhistory?from={from_str}&to={to_str}"

    try:
        result = driver.execute_async_script(f"""
            var callback = arguments[arguments.length - 1];
            fetch("{url}")
                .then(function(r) {{ return r.json(); }})
                .then(function(data) {{ callback(JSON.stringify(data)); }})
                .catch(function(e) {{ callback(JSON.stringify({{"error": e.toString()}})); }});
        """)
        import json
        data = json.loads(result)
        if "data" in data and data["data"]:
            # Save to temp CSV
            df = pd.DataFrame(data["data"])
            temp_path = os.path.join(DOWNLOAD_DIR, "vix_temp.csv")
            os.makedirs(DOWNLOAD_DIR, exist_ok=True)
            df.to_csv(temp_path, index=False)
            return temp_path
        elif "error" in data:
            print(f"   ⚠️  Script error: {data['error']}")
    except Exception as e:
        print(f"   ⚠️  Script fallback failed: {e}")

    return None


def try_scrape_table(driver):
    """Last resort: Scrape the HTML table on the page directly."""
    try:
        tables = driver.find_elements(By.TAG_NAME, "table")
        for table in tables:
            if table.is_displayed():
                rows = table.find_elements(By.TAG_NAME, "tr")
                if len(rows) > 1:
                    data = []
                    headers = [th.text.strip() for th in rows[0].find_elements(By.TAG_NAME, "th")]
                    for row in rows[1:]:
                        cells = [td.text.strip() for td in row.find_elements(By.TAG_NAME, "td")]
                        if cells:
                            data.append(cells)
                    if data:
                        df = pd.DataFrame(data, columns=headers if headers else None)
                        temp_path = os.path.join(DOWNLOAD_DIR, "vix_scraped.csv")
                        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
                        df.to_csv(temp_path, index=False)
                        return temp_path
    except Exception as e:
        print(f"   ⚠️  Table scrape failed: {e}")
    return None


# ── Main Extraction Logic ──────────────────────────────────────────────────

def extract_vix_data():
    """Main function to extract VIX historical data using Selenium UI automation."""
    print("=" * 60)
    print("  India VIX Historical Data Extraction (Selenium UI)")
    print(f"  Date Range: {START_DATE} → {END_DATE}")
    print("=" * 60)

    driver = None
    try:
        driver = create_browser()

        # First visit NSE homepage to establish cookies
        print("\n🌐 Establishing session with NSE...")
        driver.get(BASE_URL)
        wait_for_page_load(driver)
        time.sleep(3)
        print(f"✅ NSE loaded. Title: {driver.title}")

        # Generate date chunks
        chunks = generate_date_chunks(START_DATE, END_DATE)
        print(f"\n📊 Total chunks to process: {len(chunks)}\n")

        all_dfs = []
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)

        for i, (from_date, to_date) in enumerate(chunks, 1):
            print(f"[{i}/{len(chunks)}] {from_date} → {to_date}:")

            clear_download_dir()

            # Try the script-based fallback first (faster, uses browser's own session)
            csv_path = try_download_via_script(driver, from_date, to_date)

            # If script fails, try full UI automation
            if not csv_path:
                csv_path = try_download_via_ui(driver, from_date, to_date)

            if csv_path and os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    if len(df) > 0:
                        all_dfs.append(df)
                        print(f"   📋 {len(df)} rows loaded")
                    else:
                        print(f"   ⚠️  Empty CSV")
                except Exception as e:
                    print(f"   ⚠️  Error reading CSV: {e}")
            else:
                print(f"   ⚠️  No data for this chunk")

            # Delay between chunks
            if i < len(chunks):
                time.sleep(4)

            # Refresh session every 3 chunks
            if i % 3 == 0 and i < len(chunks):
                print("   🔄 Refreshing session...")
                driver.get(BASE_URL)
                wait_for_page_load(driver)
                time.sleep(2)

        if not all_dfs:
            print("\n❌ No data was retrieved!")
            print("   This may be because NSE is blocking automated access.")
            print("   You can try manually downloading from:")
            print(f"   {VIX_PAGE}")
            return None

        # Merge all dataframes
        df = pd.concat(all_dfs, ignore_index=True)
        print(f"\n📋 Total records before cleanup: {len(df)}")

        # Standardize column names
        column_mapping = {
            "EOD_TIMESTAMP": "Date",
            "EOD_OPEN_INDEX_VAL": "Open",
            "EOD_HIGH_INDEX_VAL": "High",
            "EOD_LOW_INDEX_VAL": "Low",
            "EOD_CLOSE_INDEX_VAL": "Close",
            "EOD_PREV_CLOSE": "Previous_Close",
            "EOD_CHG": "Change",
            "EOD_PERCENT_CHG": "Pct_Change",
            "TIMESTAMP": "Date",
            "VIX_OPEN": "Open",
            "VIX_HIGH": "High",
            "VIX_LOW": "Low",
            "VIX_CLOSE": "Close",
            "VIX_PREV_CLOSE": "Previous_Close",
            "CHANGE": "Change",
            "PERCHANGE": "Pct_Change",
        }

        rename_map = {k: v for k, v in column_mapping.items() if k in df.columns}
        if rename_map:
            df.rename(columns=rename_map, inplace=True)

        # Parse dates and sort
        if "Date" in df.columns:
            for fmt in ["%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d", "%b %d, %Y"]:
                try:
                    df["Date"] = pd.to_datetime(df["Date"], format=fmt)
                    break
                except (ValueError, TypeError):
                    continue
            else:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

            df.sort_values("Date", inplace=True)
            df.reset_index(drop=True, inplace=True)

        # Remove duplicates
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Save final CSV
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n💾 Data saved to: {OUTPUT_FILE}")
        print(f"   Rows: {len(df)} | Columns: {list(df.columns)}")

        # Print sample
        print("\n📄 First 5 rows:")
        print(df.head().to_string())
        print("\n📄 Last 5 rows:")
        print(df.tail().to_string())

        # Cleanup temp downloads
        clear_download_dir()
        try:
            os.rmdir(DOWNLOAD_DIR)
        except Exception:
            pass

        return df

    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        if driver:
            driver.quit()
            print("\n🔒 Browser closed.")


if __name__ == "__main__":
    extract_vix_data()
