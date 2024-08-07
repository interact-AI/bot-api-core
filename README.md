# Setup Instructions

### create virtualenv

```
python3 -m venv .venv && source .venv/bin/activate
```

### Install packages

```
pip install -r requirements.txt
```

### Environment variables

All env variables goes to .env

### Updating the requirements

[REFERENCE](https://stackoverflow.com/questions/66751657/what-does-pip-compile-do-what-is-its-use-how-do-i-maintain-the-contents-of-my)

The requirements are located in requirements.in file, update with the desired packages and run

```
    pip-compile requirements.in
```

To create requrements.txt with the specific versions of the packages required.
to do this you need to previously install pip-tools

```
    pip install pip-tools
```

### Run the python files

```
python3 ingest.py //To save pdf into vector database
python3 langgraph_bot.py //For running the agent
```

## Branching

We have created a repository on GitHub, the link to which can be found at the beginning of the report.

The branching model we will use is GitFlow, which consists of utilizing two main branches:

- **Main:** This is the primary branch that contains stable code ready for release.
- **Develop:** This branch serves as the base for software development. It is used for ongoing work and to integrate changes from various feature branches.

## Gitflow

The workflow for adding new features to the repository is as follows:

1. Create a new branch from `develop`, named according to the following format: `(change type)/(change name)`, for example: `feature/ticket-name`, `bugfix/ticket-name`, `hotfix/ticket-name`.
2. Develop the feature in this branch.

3. Pull request to `develop`, where you must briefly specify what was done in the branch and the design decisions made during these changes. Then, a code review is required from someone other than the person who made the requirement. If the changes are approved, merge them into `develop`. GitHub is configured to prevent merging pull requests that have not been approved.

# RAG-chat-with-documents

Chainlit app for advanced RAG. Uses llamaparse, langchain, qdrant and models from groq.

## Videos covering these topics

### [Llamaparse LlamaIndex](https://youtu.be/wRMnHbiz5ck?si=iQZV7N6-trcuBm8M)

### [Llamaparse Qdrant Gorq](https://youtu.be/w7Ap6gZFXl0?si=05AUGmRp1quTdeZl)

### [RAG With LlamaParse from LlamaIndex & LangChain 🚀](https://youtu.be/f9hvrqVvZl0?si=qnJBsAZD4hBUweiS)

### Links shown in video

- [LlamaCloud](https://cloud.llamaindex.ai/)
- [Qdrant Cloud](https://cloud.qdrant.io/)
- [Groq Cloud](https://console.groq.com/)

# EchoBot

Bot Framework v4 echo bot sample.

This bot has been created using [Bot Framework](https://dev.botframework.com), it shows how to create a simple bot that accepts input from the user and echoes it back.

## To try this sample

- Clone the repository

```bash
git clone https://github.com/Microsoft/botbuilder-samples.git
```

- In a terminal, navigate to `botbuilder-samples\samples\python\02.echo-bot` folder
- Activate your desired virtual environment
- In the terminal, type `pip install -r requirements.txt`
- Run your bot with `python app.py`

## Testing the bot using Bot Framework Emulator

[Bot Framework Emulator](https://github.com/microsoft/botframework-emulator) is a desktop application that allows bot developers to test and debug their bots on localhost or running remotely through a tunnel.

- Install the latest Bot Framework Emulator from [here](https://github.com/Microsoft/BotFramework-Emulator/releases)

### Connect to the bot using Bot Framework Emulator

- Launch Bot Framework Emulator
- File -> Open Bot
- Enter a Bot URL of `http://localhost:3978/api/messages`

## Interacting with the bot

Enter text in the emulator. The text will be echoed back by the bot.

## Deploy the bot to Azure

To learn more about deploying a bot to Azure, see [Deploy your bot to Azure](https://aka.ms/azuredeployment) for a complete list of deployment instructions.

## Further reading

- [Bot Framework Documentation](https://docs.botframework.com)
- [Bot Basics](https://docs.microsoft.com/azure/bot-service/bot-builder-basics?view=azure-bot-service-4.0)
- [Activity processing](https://docs.microsoft.com/en-us/azure/bot-service/bot-builder-concept-activity-processing?view=azure-bot-service-4.0)
- [Azure Bot Service Introduction](https://docs.microsoft.com/azure/bot-service/bot-service-overview-introduction?view=azure-bot-service-4.0)
- [Azure Bot Service Documentation](https://docs.microsoft.com/azure/bot-service/?view=azure-bot-service-4.0)
- [Azure CLI](https://docs.microsoft.com/cli/azure/?view=azure-cli-latest)
- [Azure Portal](https://portal.azure.com)
- [Channels and Bot Connector Service](https://docs.microsoft.com/en-us/azure/bot-service/bot-concepts?view=azure-bot-service-4.0)
