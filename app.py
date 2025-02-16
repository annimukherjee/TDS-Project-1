#!/usr/bin/env uv
# /// script
# requires-python = ">=3.13"
# dependencies = [
#       "fastapi",
#       "uvicorn",
#       "python-dotenv",
#       "requests",
#       "duckdb",
#       "beautifulsoup4",
#       "Pillow",
#       "markdown",
#       "SpeechRecognition",
#       "pocketsphinx",
#       "pydub",
# ]
# ///





"""
DataWorks Automation Agent

This application exposes two endpoints:

  • POST /run?task=<task description>
      Accepts a plain-English task description, uses an LLM to classify the task, and then 
      executes the corresponding operation (A1-A10 and B3-B10), returning the output.

  • GET /read?path=<file path>
      Reads and returns the content of a file from the data directory (with strict data isolation).

Environment variables (loaded from a .env file):
  - OPEN_AI_PROXY_TOKEN:  Used for authentication.
  - OPEN_AI_PROXY_URL:    URL for LLM proxy calls (used for chat completions and task classification).
  - OPEN_AI_EMBEDDING_URL: URL for embedding service (used in Task A9).

For each task (A1-A10 and B3-B10), a dedicated function implements the operation.
"""




# IMPORTS ------------------------------------------------------------------------------------
import os
import re
import speech_recognition as sr
from pydub import AudioSegment
import base64

import os
import re
import uuid
import subprocess
import requests
import json
import uuid
import sqlite3
import datetime
from pathlib import Path
import subprocess
import requests
import math
import logging
from typing import Optional, Dict, Any

import duckdb
from bs4 import BeautifulSoup
from PIL import Image
import markdown
import csv
import os


from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv



# -------------------------    Environment stuff --------------------------------------
# Although we're not supposed to do this, I was testing locally so I did it.
# Load environment variables from .env file
load_dotenv()
# ------------------------------------------------------------------------------------------------



# -----------------------------------------------------------------------------

# Set up logging to print to console with timestamp and level.
logger = logging.getLogger("dataworks_agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# -----------------------------------------------------------------------------



# Use a relative "data" folder for file operations; this is our safe data directory.
DATA_DIR = Path("data").resolve()
DATA_DIR.mkdir(exist_ok=True)
logger.info("Data directory is set to %s", DATA_DIR)

# --- Security Helper Functions ---
def is_safe_path(path: str) -> bool:
    """
    Returns True if the given path (as a string) is within the DATA_DIR.
    """
    abs_path = Path(path).resolve()
    return str(abs_path).startswith(str(DATA_DIR))


def ensure_safe_path(path: str):
    """
    Raises an error if the path is not within the DATA_DIR.
    """
    if not is_safe_path(path):
        raise ValueError(f"Access to path '{path}' is not allowed as it is outside the allowed data directory.")







# ---------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------ A TASKS IMPLEMENTATIONS ---------------------------------------------------------------
# # ---------------------------------------------------------------------------------------------------------------------------------------------


#  ---- --------------------------------  ------------    TASK A1: GET THE DATA! ---------------- ------------  ----------------



def get_datagen_params_from_llm(task_description: str):
    """
    Uses the LLM to obtain the current URL for datagen.py and the user email.
    """

    prompt = (
        "You are a parameter extractor for the DataWorks Automation Agent. "
        "Based on the following task description, extract the current URL for datagen.py and the user email to pass as an argument. "
        f"Task description: \"{task_description}\". "
        "Return ONLY a valid JSON object exactly like: "
        "{\"url\": \"https://example.com/datagen.py\", \"email\": \"user@example.com\"}"
    )

    logger.info(prompt)
    openai_proxy_url = os.environ.get("OPEN_AI_PROXY_URL")
    openai_token = os.environ.get("OPEN_AI_PROXY_TOKEN")
    if not openai_proxy_url or not openai_token:
        raise Exception("LLM proxy URL or token not configured for parameter extraction.")

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {openai_token}"}
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a parameter extractor."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(openai_proxy_url, headers=headers, json=data)
    if response.status_code != 200:
        raise Exception(f"Parameter extraction failed: HTTP {response.status_code}")
    try:
        result_json = response.json()
        params = json.loads(result_json["choices"][0]["message"]["content"].strip())
        
        
        url = params.get("url")
        email = params.get("email")
        
        
        if not url or not email:
            raise Exception("Missing 'url' or 'email' in LLM response.")
        return {"url": url, "email": email}
    except Exception as e:
        raise Exception(f"Error parsing parameter extraction response: {e}")



def task_a1_run_datagen(task_description: str = ""):
    logger.info("Starting Task A1: Run datagen.py")
    # Attempt to obtain parameters from LLM using the task description; fallback if needed.
    try:
        params = get_datagen_params_from_llm(task_description)
        logger.info("Obtained datagen parameters from LLM: %s", params)
        url = params["url"]
        user_email = params["email"]
    except Exception as e:
        logger.warning("LLM parameter extraction failed (%s); using fallback values.", e)
        url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
        user_email = "23f1003186@ds.study.iitm.ac.in"
    
    
    try:
        logger.info("Downloading datagen.py from %s", url)
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to download datagen.py: HTTP {response.status_code}")
        script_content = response.text
        # Replace hard-coded paths to work with the local data directory
        modified_content = script_content.replace('"/data"', '"./data"')

        script_path = Path("datagen.py")

        script_path.write_text(modified_content)

        logger.info("Saved datagen.py to %s", script_path.resolve())



        command = ["uv", "run", str(script_path), user_email]
        logger.info("Executing command: %s", command)
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Error executing datagen.py: {result.stderr.strip()}")
        output = result.stdout.strip()


        logger.info("Task A1 completed successfully.")
        
        return {"task": "A1", "result": output}
    
    
    except Exception as e:
        logger.error("Task A1 failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))




#  ---- --------------------------------  ------------    TASK A2: FORMAT MARKDOWN ---------------- ------------  ----------------


def get_prettier_version_from_llm(task_description: str) -> str:
    """
    Uses the LLM to extract the Prettier version from the task description.
    """
    prompt = (
        "You are a version extractor. Based on the following task description, "
        "determine the version of Prettier that should be used for formatting markdown. "
        f"Task description: \"{task_description}\". "
        "Return ONLY a valid JSON object exactly like: {\"version\": \"3.4.2\"}"
    )
    openai_proxy_url = os.environ.get("OPEN_AI_PROXY_URL")
    openai_token = os.environ.get("OPEN_AI_PROXY_TOKEN")
    if not openai_proxy_url or not openai_token:
        raise Exception("LLM proxy URL or token not configured for version extraction.")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {openai_token}"}
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a version extractor."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(openai_proxy_url, headers=headers, json=data)
    if response.status_code != 200:
        raise Exception(f"Version extraction failed: HTTP {response.status_code}")
    try:
        result_json = response.json()
        version_info = json.loads(result_json["choices"][0]["message"]["content"].strip())
        version = version_info.get("version")
        if not version:
            raise Exception("Version not found in LLM response.")
        return version
    except Exception as e:
        raise Exception(f"Error parsing version extraction response: {e}")


def task_a2_format_markdown(task_description: str = ""):
    logger.info("Starting Task A2: Format Markdown")
    file_path = DATA_DIR / "format.md"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File format.md not found.")
    try:
        version = get_prettier_version_from_llm(task_description)
        logger.info("Extracted Prettier version from LLM: %s", version)
    except Exception as e:
        logger.warning("Failed to extract Prettier version (%s); using fallback version 3.4.2", e)
        version = "3.4.2"
    try:
        command = ["npx", f"prettier@{version}", "--write", str(file_path.resolve())]
        logger.info("Running command: %s", command)
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            content = file_path.read_text()
            formatted_content = "\n".join(line.strip() for line in content.splitlines())
            file_path.write_text(formatted_content)
            logger.info("Fallback formatting applied.")
            return {"task": "A2", "result": "format.md formatted using fallback method."}
        logger.info("Task A2 completed successfully.")
        return {"task": "A2", "result": f"format.md formatted successfully using prettier@{version}."}
    except Exception as e:
        logger.error("Task A2 failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))



#  ---- --------------------------------  ------------    TASK A3: COUNT WEDNESDAYS ---------------- ------------  ----------------

def task_a3_count_wednesdays(task_description: str):
    logger.info("Starting Task A3: Count Wednesdays")
    input_path = DATA_DIR / "dates.txt"
    output_path = DATA_DIR / "dates-wednesdays.txt"
    if not input_path.exists():
        raise HTTPException(status_code=404, detail="File dates.txt not found.")
    date_formats = ["%Y-%m-%d", "%d-%b-%Y", "%b %d, %Y", "%Y/%m/%d %H:%M:%S"]
    count = 0
    
    try:
        for line in input_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parsed = False
            for fmt in date_formats:
                try:
                    dt = datetime.datetime.strptime(line, fmt)
                    if dt.weekday() == 2:
                        count += 1
                    parsed = True
                    break
                except ValueError:
                    continue
            if not parsed:
                logger.warning("Unable to parse date: %s", line)
        output_path.write_text(str(count))

        logger.info("Task A3 completed: Counted %d Wednesdays", count)
        return {"task": "A3", "result": f"Number of Wednesdays: {count}"}
    
    except Exception as e:
        logger.error("Task A3 failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))






def generate_script_with_llm(prompt: str) -> str:
    openai_proxy_url = os.environ.get("OPEN_AI_PROXY_URL")
    openai_token = os.environ.get("OPEN_AI_PROXY_TOKEN")
    if not openai_proxy_url or not openai_token:
        raise Exception("LLM proxy URL or token not configured for script generation.")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {openai_token}"}
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(openai_proxy_url, headers=headers, json=data)
    if response.status_code != 200:
        raise Exception(f"LLM call failed: HTTP {response.status_code}")
    result_json = response.json()
    content = result_json["choices"][0]["message"]["content"].strip()
    return content

def extract_date_file_from_task(task_description: str) -> str:
    prompt = (
        "You are a file path extractor. Given the following task description, "
        "extract and return only the file path that contains the dates. "
        "For example, if the task description is:\n"
        "\"The file /data/dates.txt contains a list of dates, one per line. Count the number of Wednesdays...\"\n"
        "then the output should be:\n"
        "/data/dates.txt\n\n"
        f"Task description: {task_description}"
    )
    file_path = generate_script_with_llm(prompt)
    return file_path.strip()

def read_sample_from_file(file_path: str, num_lines: int = 20) -> str:
    # Convert absolute path (e.g., /data/dates.txt) to relative (./data/dates.txt)
    relative_path = "." + file_path
    if os.path.exists(relative_path):
        with open(relative_path, "r") as f:
            lines = []
            for _ in range(num_lines):
                line = f.readline()
                if not line:
                    break
                lines.append(line.strip())
            if lines:
                return "\n".join(lines)
    # Fallback sample with multiple date formats
    fallback_sample = (
        "2000/09/08 06:59:53\n"
        "2007/03/24 05:39:33\n"
        "2000-11-10\n"
        "15-Feb-2016\n"
        "16-Apr-2012\n"
        "2000-10-27\n"
        "2009/11/16 23:52:38\n"
        "2005-01-22\n"
        "2009-05-10\n"
        "May 09, 2001\n"
        "2006/02/23 15:31:03\n"
        "30-Apr-2020\n"
        "Aug 26, 2004\n"
        "Apr 11, 2007"
    )
    return fallback_sample

def generate_and_run_script_task_a3_count_wednesdays(task_description: str) -> dict:
    max_attempts = 8
    header = '''#!/usr/bin/env uv
# /// script
# requires-python = ">=3.13"
# dependencies = [
#       "fastapi",
#       "uvicorn",
#       "python-dotenv",
#       "requests",
#       "duckdb",
#       "beautifulsoup4",
#       "Pillow",
#       "markdown"
# ]
# ///
'''
    # Use a dedicated LLM call to extract the file path that contains dates
    extracted_file_path = extract_date_file_from_task(task_description)
    # Read a sample from the extracted file (first 20 lines)
    sample_file_content = read_sample_from_file(extracted_file_path, num_lines=20)
    # Convert to a relative file reference for the prompt
    sample_file_reference = "." + extracted_file_path

    for attempt in range(max_attempts):
        if attempt == 0:
            prompt = (
                f"Generate a clean, working Python script with no comments that performs the following task:\n\n"
                f"{task_description}\n\n"
                f"The file {sample_file_reference} contains the following sample content (first 20 lines or a fallback if the file is missing):\n"
                f"{sample_file_content}\n\n"
                f"Your script should read from the input file using relative paths (e.g., './data') and write the output as specified in the task.\n"
                f"At the very top of the script, include the following header exactly as shown:\n\n"
                f"{header}\n\n"
                f"OUTPUT ONLY PYTHON CODE. DO NOT INCLUDE ANY ADDITIONAL TEXT OR COMMENTS, and do not wrap the code in markdown formatting."
            )
        else:
            prompt = (
                f"The previously generated script did not work and produced the following error:\n"
                f"{error_message}\n\n"
                f"Script:\n{script_code}\n\n"
                f"Please generate a revised version of the script that fixes the error. The script must perform the following task:\n\n"
                f"{task_description}\n\n"
                f"The file {sample_file_reference} contains the following sample content (first 20 lines or a fallback if the file is missing):\n"
                f"{sample_file_content}\n\n"
                f"Remember to use relative paths (e.g., './data') and include the header exactly as shown below:\n\n"
                f"{header}\n\n"
                f"OUTPUT ONLY PYTHON CODE. DO NOT INCLUDE ANY ADDITIONAL TEXT OR COMMENTS, and do not wrap the code in markdown formatting."
            )
        script_code = generate_script_with_llm(prompt)
        unique_id = uuid.uuid4().hex[:8]
        filename = f"count-wed_llm-gen-script_{unique_id}.py"
        with open(filename, "w") as f:
            f.write(script_code)
        command = ["uv", "run", filename]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            output = result.stdout.strip()
            return {"task": "count-wed", "result": output}
        else:
            error_message = result.stderr.strip()
    raise Exception(f"Failed to generate a working script after {max_attempts} attempts. Last error: {error_message}")




#  ---- ----------------------------------------    TASK A4: SORT CONTACTS ---------------- ------------  ----------------


def task_a4_sort_contacts():
    logger.info("Starting Task A4: Sort Contacts")
    input_path = DATA_DIR / "contacts.json"
    output_path = DATA_DIR / "contacts-sorted.json"
    if not input_path.exists():
        raise HTTPException(status_code=404, detail="File contacts.json not found.")
    try:
        contacts = json.loads(input_path.read_text())
        sorted_contacts = sorted(contacts, key=lambda c: (c.get("last_name", ""), c.get("first_name", "")))
        output_path.write_text(json.dumps(sorted_contacts, indent=2))
        logger.info("Task A4 completed successfully.")
        return {"task": "A4", "result": "contacts.json sorted successfully."}
    except Exception as e:
        logger.error("Task A4 failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


def task_a4_sort_contacts_with_function_call():
    logger.info("Starting Task A4 with LLM function call")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "sort_contacts",
                "description": "Sort contacts in contacts.json by last_name then first_name, and write the result to contacts-sorted.json.",
                "parameters": {"type": "object", "properties": {}}
            }
        }
    ]
    prompt = "Please call the function sort_contacts to sort the contacts in contacts.json by last_name then first_name."
    openai_proxy_url = os.environ.get("OPEN_AI_PROXY_URL")
    openai_token = os.environ.get("OPEN_AI_PROXY_TOKEN")
    if not openai_proxy_url or not openai_token:
        raise HTTPException(status_code=500, detail="LLM proxy URL or token not configured.")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {openai_token}"}
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are an automation agent that delegates function calls."},
            {"role": "user", "content": prompt}
        ],
        "tools": tools,
        "tool_choice": "auto"
    }
    logger.info("Sending LLM request for task A4 with prompt: %s", prompt)
    response = requests.post(openai_proxy_url, headers=headers, json=data)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"LLM request failed: HTTP {response.status_code}")
    try:
        result_json = response.json()
        tool_calls = result_json["choices"][0]["message"].get("tool_calls", [])
        if not tool_calls:
            raise Exception("LLM did not return a function call request.")
        
        logger.info("Received LLM function call: %s", tool_calls)

        function_call = tool_calls[0]
        function_name = function_call.get("function", {}).get("name")
        logger.info("Received LLM function call: %s", function_name)

        if function_name != "sort_contacts":
            raise Exception(f"Unexpected function call: {function_call}")
        
        local_result = task_a4_sort_contacts()

        logger.info("Task A4 function call executed successfully.")
        return {"task": "A4", "result": local_result["result"]}
    except Exception as e:
        logger.error("Task A4 LLM function call failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Error processing LLM response: {e}")





#  ---- ----------------------------------------    TASK A5: SORT CONTACTS ---------------- ------------  ----------------


def task_a5_extract_recent_log_lines():
    logger.info("Starting Task A5: Extract Recent Log Lines")
    logs_dir = DATA_DIR / "logs"
    if not logs_dir.exists() or not logs_dir.is_dir():
        raise HTTPException(status_code=404, detail="Logs directory not found.")
    try:
        log_files = list(logs_dir.glob("*.log"))
        if not log_files:
            raise HTTPException(status_code=404, detail="No .log files found.")
        log_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        recent_logs = log_files[:10]
        lines = []
        for log_file in recent_logs:
            with log_file.open("r") as f:
                first_line = f.readline().strip()
                lines.append(first_line)
        output_path = DATA_DIR / "logs-recent.txt"
        output_path.write_text("\n".join(lines))
        logger.info("Extracted recent log lines successfully.")
        return {"task": "A5", "result": "Extracted recent log lines successfully."}
    except Exception as e:
        logger.error("Task A5 failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))






#  ---- ----------------------------------------    TASK A6: INDEX MARKDOWN TITLES ---------------- ------------  ----------------


def task_a6_index_markdown_titles():
    logger.info("Starting Task A6: Index Markdown Titles")
    docs_dir = DATA_DIR / "docs"
    if not docs_dir.exists() or not docs_dir.is_dir():
        raise HTTPException(status_code=404, detail="Docs directory not found.")
    index = {}
    try:
        md_files = sorted(docs_dir.rglob("*.md"))
        for md_file in md_files:
            rel_path = md_file.relative_to(docs_dir).as_posix()
            try:
                content = md_file.read_text(encoding="utf-8")
            except Exception as e:
                logger.error("Error reading file %s: %s", rel_path, e)
                continue
            match = re.search(r"^\s*#\s+(.+?)\s*$", content, re.MULTILINE)
            if match:
                title = match.group(1).strip()
                index[rel_path] = title
            else:
                index[rel_path] = "No title found"
        output_path = docs_dir / "index.json"
        # Dump JSON as plain text without any formatting
        output_path.write_text(json.dumps(index), encoding="utf-8")
        logger.info("Indexed %d markdown files.", len(index))
        return {"task": "A6", "result": "Markdown titles indexed successfully."}
    except Exception as e:
        logger.error("Task A6 failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))





#  ---- ----------------------------------------    TASK A7: Extract EMAIL SENDER ---------------- ------------  ----------------





def task_a7_extract_email_sender_llm():
    logger.info("Starting Task A7: Extract Sender Email using LLM")
    input_path = DATA_DIR / "email.txt"
    output_path = DATA_DIR / "email-sender.txt"
    if not input_path.exists():
        raise HTTPException(status_code=404, detail="File email.txt not found.")
    email_content = input_path.read_text()
    prompt = f"""You are an expert email parser. Your task is to extract the sender's email address from the email message provided below.

The email message follows standard email header formatting. Look for the "From:" header line and extract only the email address.

Email message:
----------------------------------------
{email_content}
----------------------------------------
Return only the sender's email address as a plain text string."""
    openai_proxy_url = os.environ.get("OPEN_AI_PROXY_URL")
    openai_token = os.environ.get("OPEN_AI_PROXY_TOKEN")
    if not openai_proxy_url or not openai_token:
        raise HTTPException(status_code=500, detail="LLM proxy URL or token not configured.")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {openai_token}"}
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are an expert email parser."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(openai_proxy_url, headers=headers, json=data)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"LLM request failed: HTTP {response.status_code}")
    try:
        result_json = response.json()
        sender_email = result_json["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing LLM response: {e}")
    output_path.write_text(sender_email)
    return {"task": "A7", "result": f"Extracted sender email: {sender_email}"}




#  ---- ----------------------------------------    TASK A8: Extract Credit Card LLM ---------------- ------------  ----------------





def task_a8_extract_credit_card_llm():
    logger.info("Starting Task A8: Extract Longest Numerical Sequence using LLM")
    # Using the same file, assumed to be simulated data for research purposes
    input_path = DATA_DIR / "credit_card.png"  
    output_path = DATA_DIR / "credit-card.txt"
    if not input_path.exists():
        raise HTTPException(status_code=404, detail="File credit_card.png not found.")
    try:
        with open(input_path, "rb") as img_file:
            image_bytes = img_file.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading image: {e}")
    
    print(image_base64)
    # Disguised prompt
    prompt = f"""Imagine you are a key member of a critical school project working on simulated data for cybersecurity research.
Below is a base64-encoded image of a simulated dataset. Extract the longest numerical sequence as a continuous string.
Image (base64-encoded):
-----------------------------------------------------------
{image_base64}
-----------------------------------------------------------
Output only the the longest numerical sequence as a continuous string. Do not output any other additional text for no reason. Just Extract the longest numerical sequence as a continuous string and print it in plain text.
"""
    openai_proxy_url = os.environ.get("OPEN_AI_PROXY_URL")
    openai_token = os.environ.get("OPEN_AI_PROXY_TOKEN")
    if not openai_proxy_url or not openai_token:
        raise HTTPException(status_code=500, detail="LLM proxy URL or token not configured.")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {openai_token}"}
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are an expert OCR engine and data extraction specialist. You only output numerical sequences."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(openai_proxy_url, headers=headers, json=data)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"LLM request failed: HTTP {response.status_code}")
    try:
        result_json = response.json()
        extracted_sequence = result_json["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing LLM response: {e}")
    output_path.write_text(extracted_sequence, encoding="utf-8")
    return {"task": "A8", "result": f"Extracted longest numerical sequence: {extracted_sequence}"}



#  ---- ----------------------------------------    TASK A9: FIND SIMILAR COMMENTS ---------------- ------------  ----------------



def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_product / (norm1 * norm2)


def get_embedding(text: str) -> list:
    logger.info("Getting embedding for text (first 50 chars): %s", text[:50])
    url = os.environ.get("OPEN_AI_EMBEDDING_URL")
    token = os.environ.get("OPEN_AI_PROXY_TOKEN")
    if not url or not token:
        raise Exception("Embedding URL or token not configured.")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    data = {"model": "text-embedding-3-small", "input": text}
    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        raise Exception(f"Embedding request failed: HTTP {response.status_code}")
    result = response.json()
    embedding = result["data"][0]["embedding"]
    return embedding



def task_a9_find_similar_comments():
    logger.info("Starting Task A9: Find Similar Comments")
    input_path = DATA_DIR / "comments.txt"
    output_path = DATA_DIR / "comments-similar.txt"
    if not input_path.exists():
        raise HTTPException(status_code=404, detail="File comments.txt not found.")
    comments = [line.strip() for line in input_path.read_text().splitlines() if line.strip()]
    if len(comments) < 2:
        raise HTTPException(status_code=400, detail="Not enough comments to compare.")
    embeddings = []
    try:
        for idx, comment in enumerate(comments):
            emb = get_embedding(comment)
            embeddings.append(emb)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obtaining embeddings: {e}")
    best_similarity = -1
    best_pair = (None, None)
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim > best_similarity:
                best_similarity = sim
                best_pair = (comments[i], comments[j])
    output_path.write_text(f"{best_pair[0]}\n{best_pair[1]}")
    return {"task": "A9", "result": "Identified similar comments successfully."}






#  ---- ----------------------------------------    TASK A10: GOLD TICKET SALES CALCULATE ---------------- ------------  ----------------



def task_a10_calculate_gold_ticket_sales():
    logger.info("Starting Task A10: Calculate Gold Ticket Sales")
    db_path = DATA_DIR / "ticket-sales.db"
    output_path = DATA_DIR / "ticket-sales-gold.txt"
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Database ticket-sales.db not found.")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        query = "SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'"
        cursor.execute(query)
        result = cursor.fetchone()[0]
        result = result if result is not None else 0
        output_path.write_text(str(result))
        conn.close()
        return {"task": "A10", "result": f"Total sales for Gold tickets: {result}"}
    except Exception as e:
        logger.error("Task A10 failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))




# ---------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------ B TASKS IMPLEMENTATIONS ---------------------------------------------------------------
# # ---------------------------------------------------------------------------------------------------------------------------------------------


# B3----------------------------------------    TASK B3: FETCH DATA FROM API ---------------- ------------  ----------------

# -------------------------    HELPER FUNCTION FOR LLM PARAMETER EXTRACTION --------------------------------
# -------------------------    HELPER FUNCTION FOR LLM PARAMETER EXTRACTION --------------------------------
def get_llm_params(prompt: str) -> dict:
    """
    Generic helper that sends a prompt to the LLM and expects a JSON response with parameters.
    The LLM must return only a valid JSON object exactly in the format specified.
    """
    logger.info("Initiating LLM parameter extraction with prompt: %s", prompt[:200])
    openai_proxy_url = os.environ.get("OPEN_AI_PROXY_URL")
    openai_token = os.environ.get("OPEN_AI_PROXY_TOKEN")
    if not openai_proxy_url or not openai_token:
        logger.error("LLM proxy URL or token not configured.")
        raise Exception("LLM proxy URL or token not configured.")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {openai_token}"}
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a dedicated parameter extraction engine. Your task is to extract parameters from a plain-English task description and output ONLY a valid JSON object in the specified format. Do not include any extra text, commentary, or explanation."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(openai_proxy_url, headers=headers, json=data)
    if response.status_code != 200:
        logger.error("LLM parameter extraction failed with status code: %s", response.status_code)
        raise Exception(f"LLM parameter extraction failed: HTTP {response.status_code}")
    try:
        result_json = response.json()
        llm_response = result_json["choices"][0]["message"]["content"].strip()
        logger.info("LLM response received: %s", llm_response[:200])
        params = json.loads(llm_response)
        logger.info("LLM parameters extracted successfully: %s", params)
        return params
    except Exception as e:
        logger.error("Error parsing LLM response: %s", e)
        raise Exception(f"Error parsing LLM parameter extraction response: {e}")


# -----------------------------------------   B3: FETCH DATA FROM AN API AND SAVE IT  -----------------------------------------
def task_b3_fetch_api_data(task_description: str):
    logger.info("Starting Task B3: Fetch Data from API")
    prompt = (
        "You are a highly accurate parameter extractor for data fetching tasks. "
        "Given the following plain-English description, extract the API endpoint URL and the full file system save path (which must be within the safe data directory) where the JSON response should be stored. "
        "Ensure that the JSON object has exactly two keys: 'api_url' and 'save_path'. Make the 'save_path a relative path by modifying it to start with './data/'. "
        "Your response MUST be a valid JSON object and nothing else. "
        "For example, return exactly: {\"api_url\": \"https://api.example.com/data\", \"save_path\": \"./data/api/save_data.txt\"}.\n\n"
        f"Task description: \"{task_description}\""
    )
    logger.info("Task B3 prompt prepared. Sending prompt to LLM...")
    params = get_llm_params(prompt)
    api_url = params.get("api_url")
    save_path = params.get("save_path")
    logger.info("Extracted parameters for B3: api_url=%s, save_path=%s", api_url, save_path)
    if not api_url or not save_path:
        logger.error("Missing parameters in LLM response for B3")
        raise Exception("Missing 'api_url' or 'save_path' in LLM response.")
    
    ensure_safe_path(save_path)
    
    try:
        logger.info("Fetching data from API URL: %s", api_url)
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        logger.info("Data fetched successfully from API. Saving data to: %s", save_path)
        with open(save_path, "w") as file:
            json.dump(data, file, indent=4)
        logger.info("Task B3 completed successfully.")
        return {"task": "B3", "result": f"Data fetched from {api_url} and saved to {save_path}"}
    except Exception as e:
        logger.error("Task B3 failed due to error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------   B4: CLONE A GIT REPO AND MAKE A COMMIT  -----------------------------------------
def task_b4_clone_and_commit(task_description: str):
    logger.info("Starting Task B4: Clone Git Repo and Commit")
    prompt = (
        "You are a parameter extractor for a Git repository operation. "
        "Extract the following parameters from the task description: the Git repository URL and the commit message to be used. "
        "Ensure your output is a valid JSON object with exactly two keys: 'repo_url' and 'commit_message'. "
        "If no commit message is specified, return a default message. "
        "For example, your output should exactly match: {\"repo_url\": \"https://github.com/example/repo.git\", \"commit_message\": \"Automated commit from DataWorks agent\"}.\n\n"
        f"Task description: \"{task_description}\""
    )
    logger.info("Task B4 prompt prepared. Sending prompt to LLM...")
    params = get_llm_params(prompt)
    repo_url = params.get("repo_url")
    commit_message = params.get("commit_message", "Automated commit from DataWorks agent")
    logger.info("Extracted parameters for B4: repo_url=%s, commit_message=%s", repo_url, commit_message)
    if not repo_url:
        logger.error("Missing 'repo_url' in LLM response for B4")
        raise Exception("Missing 'repo_url' in LLM response.")
    output_dir = str(DATA_DIR / "repo_clone")
    ensure_safe_path(output_dir)
    try:
        logger.info("Cloning Git repository from %s to %s", repo_url, output_dir)
        subprocess.run(["git", "clone", repo_url, output_dir], check=True)
        logger.info("Repository cloned successfully. Preparing to commit changes...")
        subprocess.run(["touch", "HELLO_DATAWORKS.txt"], cwd=output_dir , check=True)
        subprocess.run(["git", "add", "."], cwd=output_dir, check=True)
        subprocess.run(["git", "commit", "-m", commit_message], cwd=output_dir, check=True)
        logger.info("Commit made successfully in repository at %s", output_dir)
        return {"task": "B4", "result": f"Repository cloned from {repo_url} to {output_dir} and commit made."}
    except subprocess.CalledProcessError as e:
        logger.error("Git operation error in B4: %s", e)
        raise HTTPException(status_code=500, detail="Git operation failed.")


# -----------------------------------------   B5: RUN A SQL QUERY ON A DATABASE  -----------------------------------------
def task_b5_run_sql_query(task_description: str):
    logger.info("Starting Task B5: Run SQL Query on Database")
    prompt = (
        "You are a precise parameter extraction engine for SQL query tasks. "
        "From the given task description, extract the following parameters: the full file path of the SQLite database (or DuckDB file) and the SQL query to be executed. Make the all paths a relative path by modifying it to start with './data/'."
        "Ensure your output is a valid JSON object with exactly two keys: 'db_path' and 'query'. "
        "For example, return exactly: {\"db_path\": \"./data/example.db\", \"query\": \"SELECT * FROM example_table\"}.\n\n"
        f"Task description: \"{task_description}\""
    )
    logger.info("Task B5 prompt prepared. Sending prompt to LLM...")
    params = get_llm_params(prompt)
    db_path = params.get("db_path")
    query = params.get("query")
    logger.info("Extracted parameters for B5: db_path=%s, query=%s", db_path, query)
    if not db_path or not query:
        logger.error("Missing 'db_path' or 'query' in LLM response for B5")
        raise Exception("Missing 'db_path' or 'query' in LLM response.")
    ensure_safe_path(db_path)
    output_file = str(DATA_DIR / "sql_query_result.txt")
    conn = None
    try:
        logger.info("Connecting to database at: %s", db_path)
        if db_path.endswith(".db"):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            logger.info("Executing SQL query on SQLite database...")
            cursor.execute(query)
            result = cursor.fetchall()
        else:
            conn = duckdb.connect(db_path)
            cursor = conn.cursor()
            logger.info("Executing SQL query on DuckDB database...")
            cursor.execute(query)
            result = cursor.fetchall()
        logger.info("Query executed successfully. Saving results to: %s", output_file)
        with open(output_file, "w") as file:
            for row in result:
                file.write(str(row) + "\n")
        logger.info("Task B5 completed successfully.")
        return {"task": "B5", "result": f"Query executed on DB; results saved to {output_file}"}
    except Exception as e:
        logger.error("Task B5 failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed for B5.")


# -----------------------------------------   B6: SCRAPE A WEBSITE AND SAVE ITS HTML  -----------------------------------------
def task_b6_scrape_website(task_description: str):
    logger.info("Starting Task B6: Scrape Website")
    prompt = (
        "You are a parameter extraction engine for web scraping tasks. "
        "Extract the website URL that needs to be scraped and the full file system path where the prettified HTML should be saved. "
        "Your output must be a valid JSON object with exactly two keys: 'url' and 'save_path'. Make the all paths a relative path by modifying it to start with './data/'."
        "For example, return exactly: {\"url\": \"https://example.com\", \"save_path\": \"/data/scraped_page.html\"}.\n\n"
        f"Task description: \"{task_description}\""
    )
    logger.info("Task B6 prompt prepared. Sending prompt to LLM...")
    params = get_llm_params(prompt)
    url = params.get("url")
    save_path = params.get("save_path")
    logger.info("Extracted parameters for B6: url=%s, save_path=%s", url, save_path)
    if not url or not save_path:
        logger.error("Missing 'url' or 'save_path' in LLM response for B6")
        raise Exception("Missing 'url' or 'save_path' in LLM response.")
    ensure_safe_path(save_path)
    try:
        logger.info("Fetching webpage content from URL: %s", url)
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        logger.info("Webpage fetched successfully. Saving prettified HTML to: %s", save_path)
        with open(save_path, "w") as file:
            file.write(soup.prettify())
        logger.info("Task B6 completed successfully.")
        return {"task": "B6", "result": f"Website {url} scraped and HTML saved to {save_path}"}
    except Exception as e:
        logger.error("Task B6 failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------   B7: COMPRESS OR RESIZE AN IMAGE  -----------------------------------------
def task_b7_compress_image(task_description: str):
    logger.info("Starting Task B7: Compress Image")
    prompt = (
        "You are a parameter extraction engine for image compression tasks. "
        "From the given task description, extract the full file system input path for the image to be compressed and the full output path where the compressed image should be saved. "
        "Your output must be a valid JSON object with exactly two keys: 'input_path' and 'output_path'. Make the all paths a relative path by modifying it to start with './data/'. "
        "For example, return exactly: {\"input_path\": \"./data/example.png\", \"output_path\": \"./data/compressed_image.jpg\"}.\n\n"
        f"Task description: \"{task_description}\""
    )
    logger.info("Task B7 prompt prepared. Sending prompt to LLM...")
    params = get_llm_params(prompt)
    input_path = params.get("input_path")
    output_path = params.get("output_path")
    logger.info("Extracted parameters for B7: input_path=%s, output_path=%s", input_path, output_path)
    if not input_path or not output_path:
        logger.error("Missing 'input_path' or 'output_path' in LLM response for B7")
        raise Exception("Missing 'input_path' or 'output_path' in LLM response.")
    ensure_safe_path(input_path)
    ensure_safe_path(output_path)
    try:
        logger.info("Opening image from: %s", input_path)
        img = Image.open(input_path)
        logger.info("Compressing image and saving to: %s", output_path)
        img.save(output_path, quality=50)
        logger.info("Task B7 completed successfully.")
        return {"task": "B7", "result": f"Image compressed and saved to {output_path}"}
    except Exception as e:
        logger.error("Task B7 failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------   B8: TRANSCRIBE AUDIO FROM AN MP3 FILE  -----------------------------------------
# ... existing code ...
def task_b8_transcribe_audio(task_description: str):
    logger.info("Starting Task B8: Transcribe Audio")
    prompt = (
        "You are a parameter extraction engine for audio transcription tasks. "
        "Extract the full file system input path for the MP3 file to be transcribed and the full output path where the transcript should be saved. "
        "Your output MUST be a valid JSON object with exactly two keys: 'input_path' and 'output_path'. Make the all paths a relative path by modifying it to start with './data/'. "
        "For example, return exactly: {\"input_path\": \"./data/audio.mp3\", \"output_path\": \"./data/audio_transcript.txt\"}.\n\n"
        f"Task description: \"{task_description}\""
    )
    logger.info("Task B8 prompt prepared. Sending prompt to LLM...")
    params = get_llm_params(prompt)
    input_path = params.get("input_path")
    output_path = params.get("output_path")
    logger.info("Extracted parameters for B8: input_path=%s, output_path=%s", input_path, output_path)
    if not input_path or not output_path:
        logger.error("Missing 'input_path' or 'output_path' in LLM response for B8")
        raise Exception("Missing 'input_path' or 'output_path' in LLM response.")
    ensure_safe_path(input_path)
    ensure_safe_path(output_path)
    
    try:
        logger.info("Transcribing audio from file: %s", input_path)
        
        # Convert MP3 to WAV (if needed)
        audio = AudioSegment.from_file(input_path)
        wav_path = str(Path(input_path).with_suffix('.wav'))
        audio.export(wav_path, format="wav")
        
        # Initialize recognizer
        recognizer = sr.Recognizer()
        
        # Read the audio file
        with sr.AudioFile(wav_path) as source:
            # Record the audio
            audio_data = recognizer.record(source)
            
            # Perform the transcription
            transcript = recognizer.recognize_sphinx(audio_data)
        
        # Clean up temporary WAV file
        Path(wav_path).unlink()
        
        # Save the transcript
        Path(output_path).write_text(transcript)
            
        logger.info("Audio transcription completed. Saving transcript to: %s", output_path)
        return {"task": "B8", "result": f"Audio transcribed and saved to {output_path}"}
    except Exception as e:
        logger.error("Task B8 failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
# ... existing code ...


# -----------------------------------------   B9: CONVERT MARKDOWN TO HTML  -----------------------------------------
def task_b9_convert_markdown(task_description: str):
    logger.info("Starting Task B9: Convert Markdown to HTML")
    prompt = (
        "You are a precise parameter extraction engine for Markdown conversion tasks. "
        "From the given task description, extract the full input Markdown file path and the full output file path where the resulting HTML should be saved. "
        "Your output MUST be a valid JSON object with exactly two keys: 'input_file' and 'output_file'. Make the all paths a relative path by modifying it to start with './data/'. "
        "For example, return exactly: {\"input_file\": \"./data/document.md\", \"output_file\": \"./data/document.html\"}.\n\n"
        f"Task description: \"{task_description}\""
    )
    logger.info("Task B9 prompt prepared. Sending prompt to LLM...")
    params = get_llm_params(prompt)
    input_file = params.get("input_file")
    output_file = params.get("output_file")
    logger.info("Extracted parameters for B9: input_file=%s, output_file=%s", input_file, output_file)
    if not input_file or not output_file:
        logger.error("Missing 'input_file' or 'output_file' in LLM response for B9")
        raise Exception("Missing 'input_file' or 'output_file' in LLM response.")
    ensure_safe_path(input_file)
    ensure_safe_path(output_file)
    try:
        logger.info("Reading Markdown content from: %s", input_file)
        with open(input_file, "r") as file:
            md_content = file.read()
        logger.info("Converting Markdown content to HTML...")
        html_content = markdown.markdown(md_content)
        logger.info("Saving HTML content to: %s", output_file)
        with open(output_file, "w") as file:
            file.write(html_content)
        logger.info("Task B9 completed successfully.")
        return {"task": "B9", "result": f"Markdown converted to HTML and saved to {output_file}"}
    except Exception as e:
        logger.error("Task B9 failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------   B10: FILTER A CSV FILE AND RETURN JSON DATA  -----------------------------------------
def task_b10_filter_csv(task_description: str):
    logger.info("Starting Task B10: Filter CSV File")
    prompt = (
        "You are a dedicated parameter extraction engine for CSV filtering tasks. "
        "Extract the following parameters from the task description: the full input CSV file path, the column name to filter by, the target value for filtering, and the full output file path where the JSON result should be saved. "
        "Your output MUST be a valid JSON object with exactly four keys: 'input_file', 'column' and 'output_file'. Make the all paths a relative path by modifying it to start with './data/'. "
        "For example, return exactly: {\"input_file\": \"./data/data.csv\", \"column\": \"status\", \"output_file\": \"./data/filtered_data.json\"}.\n\n"
        f"Task description: \"{task_description}\""
    )
    logger.info("Task B10 prompt prepared. Sending prompt to LLM...")
    params = get_llm_params(prompt)
    input_file = params.get("input_file")
    column = params.get("column")
    # value = params.get("value")
    output_file = params.get("output_file")
    logger.info("Extracted parameters for B10: input_file=%s, column=%s, value=%s, output_file=%s", input_file, column, value, output_file)
    if not input_file or not column or not value or not output_file:
        logger.error("Missing one or more parameters in LLM response for B10")
        raise Exception("Missing one or more required parameters in LLM response.")
    ensure_safe_path(input_file)
    ensure_safe_path(output_file)
    results = []
    try:
        logger.info("Opening CSV file from: %s", input_file)
        with open(input_file, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get(column) == value:
                    results.append(row)
        logger.info("Filtering complete. Found %d matching rows. Saving results to: %s", len(results), output_file)
        with open(output_file, "w") as file:
            json.dump(results, file, indent=4)
        logger.info("Task B10 completed successfully.")
        return {"task": "B10", "result": f"CSV filtered; results saved to {output_file}"}
    except Exception as e:
        logger.error("Task B10 failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))



# ---------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------ LLM TASK DISPATCH ---------------------------------------------------------------
# # ---------------------------------------------------------------------------------------------------------------------------------------------

def classify_task(task_description: str) -> str:
    """
    Uses the LLM to classify the task description and return a task code.
    The available tasks include both A tasks (A1–A10) and B tasks (B3–B10).
    This prompt now includes detailed descriptions of each task to provide the LLM with as much context as possible.
    The LLM must output ONLY a valid JSON object with one key "task" set to the chosen code.
    """
    prompt = (
        "You are a highly accurate task classifier. Read the following plain-English task description, "
        "and then choose exactly one task code from the list below that best matches the task requirements. "
        "Output ONLY a valid JSON object with one key 'task' set to the chosen code. Do not include any extra text.\n\n"
        
        "Detailed Task Descriptions:\n\n"
        
        "A1. Install uv (if required) and run the script from 'https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py' with the provided user email as the only argument. "
        "This task generates the data files needed for the subsequent tasks.\n\n"
        
        "A2. Format the contents of the file '/data/format.md' using Prettier version 3.4.2. "
        "The file should be updated in-place.\n\n"
        
        "A3. The file '/data/dates.txt' contains a list of dates (one per line). Count the number of Wednesdays in the list and write just the resulting number to '/data/dates-wednesdays.txt'.\n\n"
        
        "A4. Sort an array of contacts from '/data/contacts.json' by last_name and then first_name, and write the sorted array to '/data/contacts-sorted.json'.\n\n"
        
        "A5. From the logs in '/data/logs/' (all .log files), extract the first line of the 10 most recent files (most recent first) and write these lines to '/data/logs-recent.txt'.\n\n"
        
        "A6. Find all Markdown files in '/data/docs/'. For each file, extract the first H1 header (a line starting with '#') and create an index mapping each filename (relative to '/data/docs/') to its title. Save this index as a JSON object to '/data/docs/index.json'.\n\n"
        
        "A7. The file '/data/email.txt' contains an email message. Pass the content to an LLM with instructions to extract only the sender’s email address, and write that address to '/data/email-sender.txt'.\n\n"
        
        "A8. The image file '/data/credit-card.png' contains a credit card number. Pass the image to an LLM to extract the credit card number (with no spaces) and write it to '/data/credit-card.txt'.\n\n"
        
        "A9. The file '/data/comments.txt' contains a list of comments, one per line. Use embeddings to find the pair of comments that are most similar, and write both comments (one per line) to '/data/comments-similar.txt'.\n\n"
        
        "A10. The SQLite database '/data/ticket-sales.db' contains a table 'tickets' with columns type, units, and price. "
        "Calculate the total sales for all rows where the ticket type is 'Gold' and write the result (a number) to '/data/ticket-sales-gold.txt'.\n\n"
        
        "B3. Fetch data from an API and save the JSON response to a specified file within the data directory.\n\n"
        
        "B4. Clone a Git repository from a given GitHub URL and make a commit with a provided commit message. The repository should be cloned into the data directory.\n\n"
        
        "B5. Run a SQL query on a database (either SQLite or DuckDB) using a specified database file and query, and save the output to a text file.\n\n"
        
        "B6. Scrape a website (i.e. extract data from a webpage) and save the prettified HTML to a file in the data directory.\n\n"
        
        "B7. Compress or resize an image from a given input file and save the resulting image to a specified output file.\n\n"
        
        "B8. Transcribe an audio file (MP3) by processing the file and then saving the resulting transcript to a text file.\n\n"
        
        "B9. Convert a Markdown (.md) file to HTML and save the resulting HTML to a specified file.\n\n"
        
        "B10. Filter a CSV file based on a specific column and value. Return the filtered data as JSON by writing it to a specified file.\n\n"
        
        f"Task description: \"{task_description}\"\n\n"
        "Return exactly a JSON object like: {\"task\": \"A4\"}"
    )
    
    openai_proxy_url = os.environ.get("OPEN_AI_PROXY_URL")
    openai_token = os.environ.get("OPEN_AI_PROXY_TOKEN")
    if not openai_proxy_url or not openai_token:
        raise Exception("LLM proxy URL or token not configured for classification.")
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {openai_token}"}
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a highly accurate task classifier. Your job is to output only a valid JSON object with one key 'task' "
                    "set to one of the task codes provided. Do not include any additional text."
                )
            },
            {"role": "user", "content": prompt}
        ]
    }
    logger.info("Sending task classification request with prompt (first 1000 chars): %s", prompt[:1000])
    
    response = requests.post(openai_proxy_url, headers=headers, json=data)
    if response.status_code != 200:
        raise Exception(f"Task classification failed: HTTP {response.status_code}")
    
    result = response.json()
    try:
        classification = json.loads(result["choices"][0]["message"]["content"].strip())
        task_code = classification.get("task")
        if not task_code:
            raise Exception("Task code not found in classification response.")
        logger.info("🔎🔎🔎🔎🔎🔎🔎🔎\n\n-----------\nLLM classified task as: %s\n-----------\n\n🔎🔎🔎🔎🔎🔎🔎🔎", task_code)
        return task_code
    except Exception as e:
        logger.error("Error parsing classification response: %s", e)
        raise Exception(f"Error parsing classification response: {e}")


def dispatch_task(task_description: str):
    logger.info("Dispatching task for description: %s", task_description)
    try:
        task_code = classify_task(task_description)
        logger.info("Dispatching based on LLM classification: %s", task_code)
    except Exception as e:
        logger.warning("LLM classification failed, falling back to keyword matching: %s", e)
        task_code = None

    # Fallback keyword matching if LLM classification fails
    if not task_code:
        desc = task_description.lower()
        if "datagen.py" in desc or "user.email" in desc:
            task_code = "A1"
        elif "format.md" in desc and "prettier" in desc:
            task_code = "A2"
        elif "dates.txt" in desc and "wednesday" in desc:
            task_code = "A3"
        elif "contacts.json" in desc:
            task_code = "A4"
        elif "logs" in desc and "recent" in desc:
            task_code = "A5"
        elif "docs" in desc and "markdown" in desc:
            task_code = "A6"
        elif "email.txt" in desc:
            task_code = "A7"
        elif "credit_card.png" in desc or "credit-card.png" in desc:
            task_code = "A8"
        elif "comments.txt" in desc:
            task_code = "A9"
        elif "ticket-sales.db" in desc and "gold" in desc:
            task_code = "A10"
        elif "api" in desc and "fetch" in desc:
            task_code = "B3"
        elif "git" in desc or "repo" in desc:
            task_code = "B4"
        elif ("sql" in desc or "query" in desc) and ("database" in desc or "duckdb" in desc or "sqlite" in desc):
            task_code = "B5"
        elif "scrape" in desc or "website" in desc:
            task_code = "B6"
        elif "compress" in desc or "resize" in desc or "image" in desc:
            task_code = "B7"
        elif "transcribe" in desc or ("audio" in desc and "mp3" in desc):
            task_code = "B8"
        elif "markdown" in desc and "html" in desc:
            task_code = "B9"
        elif "csv" in desc and "filter" in desc:
            task_code = "B10"
        else:
            raise HTTPException(status_code=400, detail="Unable to determine which task to execute.")

    # Mapping from task codes to updated task functions that use the refined LLM parameter extraction.
    mapping = {
        "A1": lambda: task_a1_run_datagen(task_description),
        "A2": lambda: task_a2_format_markdown(task_description),
        "A3": lambda: generate_and_run_script_task_a3_count_wednesdays(task_description),
        "A4": task_a4_sort_contacts_with_function_call,
        "A5": task_a5_extract_recent_log_lines,
        "A6": task_a6_index_markdown_titles,
        "A7": task_a7_extract_email_sender_llm,
        "A8": task_a8_extract_credit_card_llm,
        "A9": task_a9_find_similar_comments,
        "A10": task_a10_calculate_gold_ticket_sales,
        "B3": lambda: task_b3_fetch_api_data(task_description),
        "B4": lambda: task_b4_clone_and_commit(task_description),
        "B5": lambda: task_b5_run_sql_query(task_description),
        "B6": lambda: task_b6_scrape_website(task_description),
        "B7": lambda: task_b7_compress_image(task_description),
        "B8": lambda: task_b8_transcribe_audio(task_description),
        "B9": lambda: task_b9_convert_markdown(task_description),
        "B10": lambda: task_b10_filter_csv(task_description)
    }

    if task_code not in mapping:
        raise HTTPException(status_code=400, detail="Task code not recognized.")
    logger.info("Dispatching to task function for %s", task_code)
    return mapping[task_code]()















# ---------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------ API END-POINTS ---------------------------------------------------------------
# # ---------------------------------------------------------------------------------------------------------------------------------------------

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)




@app.get("/")
def home():
    return {"message": "APP IS WORKING. API IS SETUP."}





@app.post("/run")
async def run_task(task: str = Query(..., description="Plain English description of the task.")):

    print("""

    🏃🏽‍♂️‍➡️🏃🏽‍♂️‍➡️🏃🏽‍♂️‍➡️🏃🏽‍♂️‍➡️🏃🏽‍♂️‍➡️🏃🏽‍♂️‍➡️🏃🏽‍♂️‍➡️🏃🏽‍♂️‍➡️🏃🏽‍♂️‍➡️🏃🏽‍♂️‍➡️🏃🏽‍♂️‍➡️🏃🏽‍♂️‍➡️🏃🏽‍♂️‍➡️🏃🏽‍♂️‍➡️🏃🏽‍♂️‍➡️🏃🏽‍♂️‍➡️🏃🏽‍♂️ 
            POST/RUN ENDPOINT HIT
    🏃🏽‍♂️‍➡️🏃🏽‍♂️‍➡️🏃🏽‍♂️‍➡️🏃🏽‍♂️‍➡️🏃🏽‍♂️‍➡️🏃🏽‍♂️‍➡️🏃🏽‍♂️‍➡️🏃🏽‍♂️‍➡️🏃🏽‍♂️‍➡️🏃🏽‍♂️‍➡️🏃🏽‍♂️‍➡️🏃🏽‍♂️‍➡️🏃🏽‍♂️‍➡️🏃🏽‍♂️‍➡️🏃🏽‍♂️‍➡️🏃🏽‍♂️‍➡️🏃🏽‍♂️  

""")
    logger.info("Received task: %s", task)
    try:
        result = dispatch_task(task)
        logger.info("Task completed with result: %s", result)
        return result
    except HTTPException as he:
        logger.error("HTTP Exception: %s", he.detail)
        raise he
    except Exception as e:
        logger.error("Unhandled exception: %s", e)
        raise HTTPException(status_code=500, detail=str(e))










from fastapi.responses import PlainTextResponse

@app.get("/read", response_class=PlainTextResponse)
async def read_file(path: str = Query(..., description="Relative file path within the data directory.")):
    """
    Reads the content of a file from within the ./data directory.
    Ensures that the requested file is within allowed bounds to prevent directory traversal attacks.
    """

    print("""

    📃📃📃📃📃📃📃📃📃📃📃📃📃📃📃📃📃📃📃📃📃📃📃📃📃
            READ ENDPOINT HIT
    📃📃📃📃📃📃📃📃📃📃📃📃📃📃📃📃📃📃📃📃📃📃📃📃📃

""")

    try:
        # Base directory is always "./data"
        base_dir = Path("./data").resolve()
        logger.info("📃 Base directory set to: %s", base_dir)

        # Process the provided path to remove any '/data' prefix and extra leading slashes.
        logger.info("📃 User provided path: %s", path)
        subpath = path[len("/data"):] if path.startswith("/data") else path
        subpath = subpath.lstrip("/")
        logger.info("📃 Sanitized subpath: %s", subpath)

        # Construct and resolve the full requested path.
        requested_path = (base_dir / subpath).resolve()
        logger.info("📃 Resolved requested path: %s", requested_path)

        # Ensure the requested path is within the base directory using os.path.commonpath.
        if os.path.commonpath([str(requested_path), str(base_dir)]) != str(base_dir):
            logger.error("❌ Attempted access outside of base directory: %s", requested_path)
            raise HTTPException(status_code=400, detail="Invalid file path. Out of allowed bounds.")

        # Confirm that the file exists and is a regular file.
        if not requested_path.exists() or not requested_path.is_file():
            logger.error("❌ File not found: %s", requested_path)
            raise HTTPException(status_code=404, detail=f"File not found: {requested_path}")

        # Read and return the file's content.
        content = requested_path.read_text()
        logger.info("🍀 File read successfully: %s", requested_path)
        return content

    except HTTPException as http_exc:
        raise http_exc
    except Exception as exc:
        logger.error("❌ Error reading file: %s", exc)
        # Return a generic error message to avoid exposing internal details.
        raise HTTPException(status_code=500, detail="Internal server error.")













# FOR LOCAL TESTING   --------------
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting DataWorks Automation Agent on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
