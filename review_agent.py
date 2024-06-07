import requests
import re
from flask import Flask, request, jsonify
import requests
import json, os
from dotenv import load_dotenv

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

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_OWNER = "KirtiJha"
REPO_NAME = "GitPal"


def get_commit_details(commit_sha):
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/commits/{commit_sha}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    response = requests.get(url, headers=headers)
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
        # Threshold is set to very low level to flag everything (testing purposes)
        # or set to True to enable HAP with default settings
        hap=ModerationHAP(input=ModerationHAPInput(enabled=True, threshold=0.01))
    ),
)


def review_code(files_list):
    from langchain_core.messages import HumanMessage, SystemMessage
    import pprint

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

  Format the response in a valid JSON format as a list of feedbacks, where the value is an object containing the filename ("fileName"),  risk score ("riskScore") and the feedback ("details"). The schema of the JSON feedback object must be:
  {
    "fileName": {
      "type": "string"
    },
    "riskScore": {
      "type": "number"
    },
    "details": {
      "type": "string"
    }
  }

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
    match = re.search(r"```(.*?)```", feedback_string, re.DOTALL)
    if match:
        json_string = match.group(1).strip()
        return json.loads(json_string)
    return []


def format_feedback_as_markdown(feedback):
    markdown_comment = "Here is the review of the code changes:\n\n"
    for item in feedback:
        markdown_comment += f"### File: {item['fileName']}\n"
        markdown_comment += f"- **Risk Score**: {item['riskScore']}\n"
        markdown_comment += f"- **Details**: {item['details']}\n\n"
    markdown_comment += "Let me know if you'd like me to review more code changes!"
    return markdown_comment


def post_review_comment(body, commit_sha):
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/commits/{commit_sha}/comments"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    data = {"body": body}
    response = requests.post(url, json=data, headers=headers)
    return response.json()


@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json
    if "commits" in data:
        for commit in data["commits"]:
            commit_sha = commit["id"]
            commit_details = get_commit_details(commit_sha)
            diff = commit_details["files"]
            files_list = [
                {"filename": file["filename"], "patch": file["patch"]} for file in diff
            ]
            feedback_string = review_code(files_list)
            print(f"feedback string - {feedback_string}")
            feedback = extract_json_from_review(feedback_string)
            print(f"feedback - {feedback}")
            review_comment = format_feedback_as_markdown(feedback)
            print(f"review comment - {review_comment}")
            # pr_number = data['pull_request']['number']
            post_review_comment(review_comment, commit_sha)
    return jsonify({"status": "reviewed"})


if __name__ == "__main__":
    app.run(port=8000, debug=True)
