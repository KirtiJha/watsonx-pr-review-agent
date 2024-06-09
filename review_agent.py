import requests
import re
from flask import Flask, request, jsonify
import json
import os
from dotenv import load_dotenv
import jwt
import time
from github import Github, GithubIntegration

from genai import Client, Credentials
from genai.extensions.langchain import LangChainEmbeddingsInterface
from genai.schema import TextEmbeddingParameters
from genai.extensions.langchain.chat_llm import LangChainChatInterface
from genai.schema import (
    DecodingMethod,
    ModerationHAP,
    ModerationHAPInput,
    ModerationParameters,
    TextGenerationParameters,
    TextGenerationReturnOptions,
)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

app = Flask(__name__)

github_app_id = "916458"

with open(
    os.path.normpath(
        os.path.expanduser(
            "~/Documents/certs/github/isc-pr-review-agent.2024-06-08.private-key.pem"
        )
    ),
    "r",
) as cert_file:
    app_key = cert_file.read()

# Create an GitHub integration instance
git_integration = GithubIntegration(
    github_app_id,
    app_key,
)

REPO_OWNER = "KirtiJha"
REPO_NAME = "watsonx-pr-review-agent"


def create_jwt(app_id, app_key):
    payload = {
        "iat": int(time.time()),
        "exp": int(time.time()) + (10 * 60),
        "iss": app_id,
    }
    token = jwt.encode(payload, app_key, algorithm="RS256")
    return token


def get_installation_token(app_id, app_key, repo_owner, repo_name):
    jwt_token = create_jwt(app_id, app_key)
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Accept": "application/vnd.github+json",
    }
    url = f"https://api.github.com/app/installations"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    installations = response.json()
    for installation in installations:
        if installation["account"]["login"] == repo_owner:
            installation_id = installation["id"]
            token_url = f"https://api.github.com/app/installations/{installation_id}/access_tokens"
            token_response = requests.post(token_url, headers=headers)
            token_response.raise_for_status()
            return token_response.json()["token"]
    raise Exception("Installation not found")


def get_pull_request_files(pr_number, installation_token):
    url = (
        f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{pr_number}/files"
    )
    headers = {"Authorization": f"Bearer {installation_token}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


credentials = Credentials(
    api_key="pak-VcI5V05aaf42FbLqO7Fc6FR7KFSz2Gmap97DO1y3-I8",
    api_endpoint="https://bam-api.res.ibm.com/v2/text/chat?version=2024-03-19",
)
client = Client(credentials=credentials)
llm = LangChainChatInterface(
    client=client,
    model_id="meta-llama/llama-3-70b-instruct",
    parameters=TextGenerationParameters(
        decoding_method=DecodingMethod.GREEDY,
        max_new_tokens=2040,
        min_new_tokens=10,
        temperature=0.2,
        top_k=40,
        top_p=0.9,
        return_options=TextGenerationReturnOptions(input_text=False, input_tokens=True),
    ),
    moderations=ModerationParameters(
        hap=ModerationHAP(input=ModerationHAPInput(enabled=True, threshold=0.01))
    ),
)


def review_code(files_list):
    from langchain_core.messages import HumanMessage, SystemMessage

    prompt = f"Review the following code changes:\n{files_list}"
    result = llm.generate(
        messages=[
            [
                SystemMessage(
                    content="""You are an expert python developer, your task is to review a set of commits.
  You are given a list of filenames and their partial contents, but note that you might not have the full context of the code.

  Only review lines of code which have been changed (added or removed) in the commit. The code looks similar to the output of a git diff command. Lines which have been removed are prefixed with a minus (-) and lines which have been added are prefixed with a plus (+). Other lines are added to provide context but should be ignored in the review.

  Begin your review by evaluating the changed code using a risk score similar to a LOGAF score but measured from 1 to 5, where 1 is the lowest risk to the code base if the code is merged and 5 is the highest risk which would likely break something or be unsafe.

  In your feedback, focus on highlighting potential bugs, improving readability if it is a problem, making code cleaner, and maximising the performance of the programming language. Flag any API keys or secrets present in the code in plain text immediately as highest risk. Rate the changes based on SOLID principles if applicable.

  Do not comment on breaking functions down into smaller, more manageable functions unless it is a huge problem. Also be aware that there will be libraries and techniques used which you are not familiar with, so do not comment on those unless you are confident that there is a problem.

  Use markdown formatting for the feedback details. Also do not include the filename or risk level in the feedback details.

  Ensure the feedback details are brief, concise, accurate. If there are multiple similar issues, only comment on the most critical.

  Include brief example code snippets in the feedback details for your suggested changes when you're confident your suggestions are improvements. Use the same programming language as the file under review.
  If there are multiple improvements you suggest in the feedback details, use an ordered list to indicate the priority of the changes.

  Format the response in a valid JSON format as a list of feedbacks:
  [
    {
      "fileName": "string",
      "line": "number",
      "riskScore": "number",
      "details": "string",
      "suggestedCode": "string"
    }
  ]

  The filenames and file contents to review are provided below as a list of JSON objects:
  """,
                ),
                HumanMessage(content=prompt),
            ]
        ],
    )

    response = result.generations[0][0].text
    return response


def extract_json_from_review(feedback_string):
    try:
        json_data = re.search(r"\[\s*{.*}\s*\]", feedback_string, re.DOTALL).group(0)
        # Remove any non-printable control characters
        json_data = re.sub(r"[\x00-\x1F\x7F]", "", json_data)
        return json.loads(json_data)
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Failed to decode JSON from feedback string: {e}")
        return []


# def format_feedback_as_markdown(feedback):
#     markdown_comment = "Here is the review of the code changes:\n\n"
#     for item in feedback:
#         markdown_comment += f"### File: {item['fileName']}\n"
#         markdown_comment += f"- **Risk Score**: {item['riskScore']}\n"
#         markdown_comment += f"- **Details**: {item['details']}\n\n"
#     markdown_comment += "Let me know if you'd like me to review more code changes!"
#     return markdown_comment


# def post_review_comment(body, pr_number, installation_token):
#     url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/issues/{pr_number}/comments"
#     headers = {"Authorization": f"Bearer {installation_token}"}
#     data = {"body": body}
#     response = requests.post(url, json=data, headers=headers)
#     response.raise_for_status()
#     return response.json()


def get_pr_diff(pr_number, installation_token):
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{pr_number}"
    headers = {
        "Authorization": f"Bearer {installation_token}",
        "Accept": "application/vnd.github.v3.diff",
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.text


def get_commit_id(pr_number, installation_token):
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{pr_number}/commits"
    headers = {"Authorization": f"Bearer {installation_token}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    commits = response.json()
    # Return the last commit id (most recent)
    return commits[-1]["sha"]


def find_position_in_diff(diff, file_name, line_number):
    position = 0
    file_diff_started = False
    for line in diff.splitlines():
        if line.startswith(f"diff --git a/{file_name} b/{file_name}"):
            file_diff_started = True
        elif file_diff_started and line.startswith("@@ "):
            # Position header
            hunk_header = line
            # Extract the starting line number for the changes
            start_line_number = int(hunk_header.split(" ")[2].split(",")[0][1:])
            if (
                start_line_number
                <= line_number
                < start_line_number + int(hunk_header.split(" ")[2].split(",")[1])
            ):
                return position
        position += 1
    return None


def post_line_comment(
    pr_number, installation_token, file_name, line_number, comment, suggested_code
):
    # Get the commit ID and diff
    commit_id = get_commit_id(pr_number, installation_token)
    diff = get_pr_diff(pr_number, installation_token)

    # Find the position in the diff
    position = find_position_in_diff(diff, file_name, line_number)

    if position is None:
        print(f"Could not find the position for {file_name} at line {line_number}")
        return

    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{pr_number}/comments"
    headers = {"Authorization": f"Bearer {installation_token}"}

    # Build the comment body with a diff suggestion if applicable
    body = f"BOT_COMMENT: {comment}"
    if suggested_code:
        body += f"\n\n```suggestion\n{suggested_code}\n```"

    data = {
        "body": body,
        "commit_id": commit_id,
        "path": file_name,
        "position": position,
    }
    print(f"Posting comment: {data}")
    response = requests.post(url, json=data, headers=headers)
    print(f"Response: {response.status_code} - {response.text}")
    response.raise_for_status()
    return response.json()


@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json

    # Ignore bot comments to prevent recursion
    if "comment" in data and "body" in data["comment"]:
        if data["comment"]["body"].startswith("BOT_COMMENT:"):
            print("Ignoring bot's own comment to prevent recursion.")
            return jsonify({"status": "ignored bot comment"})

    installation_token = get_installation_token(
        github_app_id, app_key, REPO_OWNER, REPO_NAME
    )
    if "pull_request" in data:
        pr_number = data["pull_request"]["number"]
        pr_files = get_pull_request_files(pr_number, installation_token)
        files_list = [
            {"filename": file["filename"], "patch": file["patch"]} for file in pr_files
        ]
        feedback_string = review_code(files_list)
        print(f"feedback string - {feedback_string}")
        feedback = extract_json_from_review(feedback_string)
        print(f"feedback - {feedback}")
        if feedback:
            for item in feedback:
                post_line_comment(
                    pr_number,
                    installation_token,
                    item["fileName"],
                    item["line"],
                    item["details"],
                    item.get("suggestedCode", ""),
                )
        else:
            print("No feedback to post.")
    return jsonify({"status": "reviewed"})


if __name__ == "__main__":
    app.run(port=8000, debug=True)
