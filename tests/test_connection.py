from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage

from utils.token_logger import count_tokens, log_tokens
from utils.config import CHAT_DEPLOYMENT_NAME, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, \
    EMBEDDING_DEPLOYMENT_NAME, AZURE_OPENAI_API_KEY
# Load environment variables

# Initialize the Azure Chat model via LangChain
chat = AzureChatOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=CHAT_DEPLOYMENT_NAME,
    temperature=0
)


# Create a prompt template
prompt_template = ChatPromptTemplate.from_template("{input}")


def test_azure_connection(user_input: str):
    # Format prompt
    formatted_prompt = prompt_template.format(input=user_input)

    # Count & log input tokens
    input_tokens = count_tokens(user_input, model="gpt-4")
    log_tokens(input_tokens, category="chat")
    print(f"🔢 Prompt token count: {input_tokens}")

    # Send message to Azure OpenAI
    messages = [HumanMessage(content=formatted_prompt)]
    response = chat.invoke(messages)

    # Count & log output tokens
    output_text = response.content
    output_tokens = count_tokens(output_text, model="gpt-4")
    log_tokens(output_tokens, category="chat")
    print(f"🧾 Completion token count: {output_tokens}")
    print(f"📊 Total tokens this run: {input_tokens + output_tokens}")

    return output_text


if __name__ == "__main__":
    user_input = "Hello!"
    reply = test_azure_connection(user_input)
    print("✅ Azure Chat Response:\n", reply)
