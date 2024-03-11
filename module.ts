// Import required dependencies from LangChain, dotenv, and Zod for configuration and validation
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
  HumanMessage,
  AIMessage,
} from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createRetrievalChain, createHistoryAwareRetriever, createStuffDocumentsChain } from "langchain/chains";
import { createRetrieverTool } from "langchain/tools/retriever";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { pull } from "langchain/hub";
import { createOpenAIFunctionsAgent, AgentExecutor } from "langchain/agents";
import { DynamicStructuredTool } from "@langchain/core/tools";
import { ZodObject, ZodString } from "zod";
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

/**
 * Initializes the OpenAI Chat Model with configuration.
 */
function initializeChatModel() {
  return new ChatOpenAI({});
}

/**
 * Initializes the search tool with custom configurations or defaults.
 */
function initializeSearchTool() {
  return new TavilySearchResults();
}

/**
 * Creates a prompt template for generating history-aware queries.
 */
function initializePrompts() {
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", "You are a world-class technical documentation writer."],
    ["user", "{input}"],
  ]);

  const historyAwarePrompt = ChatPromptTemplate.fromMessages([
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
    [
      "user",
      "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
    ],
  ]);

  return { prompt, historyAwarePrompt };
}

/**
 * Fetches context from a predefined URL using CheerioWebBaseLoader.
 * Splits the fetched documents into manageable chunks for processing.
 */
async function fetchContext() {
  const loader = new CheerioWebBaseLoader("https://js.langchain.com/docs/get_started/quickstart");
  const docs = await loader.load();
  console.log(docs.length, docs[0].pageContent.length);

  const splitter = new RecursiveCharacterTextSplitter();
  return splitter.splitDocuments(docs);
}

/**
 * Constructs the retrieval chain using the provided document and vector store.
 */
async function buildRetrievalChain(chatModel: ChatOpenAI, splitDocs: ReturnType<typeof fetchContext>) {
  const embeddings = new OpenAIEmbeddings();
  const vectorstore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);
  const retriever = vectorstore.asRetriever();

  const prompt = ChatPromptTemplate.fromTemplate(
    `Answer the following question based only on the provided context:

    <context>
    {context}
    </context>
    
    Question: {input}`
  );

  const documentChain = await createStuffDocumentsChain({
    llm: chatModel,
    prompt,
  });

  return createRetrievalChain({
    combineDocsChain: documentChain,
    retriever,
  });
}

/**
 * Sets up tools and agents for processing queries with langchain.
 */
async function setupLangchainTools(chatModel: ChatOpenAI) {
  const searchTool = initializeSearchTool();
  const retrieverTool = await createRetrieverTool(searchTool.asRetriever(), {
    name: "langsmith_search",
    description: "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
  });

  return [retrieverTool, searchTool];
}

/**
 * Retrieves and configures the agent for execution.
 */
async function configureAgent(tools: ReturnType<typeof setupLangchainTools>, chatModel: ChatOpenAI) {
  const agentPrompt = await pull<ChatPromptTemplate>("hwchase17/openai-functions-agent");
  const agentModel = new ChatOpenAI({ modelName: "gpt-3.5-turbo-1106", temperature: 0 });

  const agent = await createOpenAIFunctionsAgent({
    llm: agentModel,
    tools,
    prompt: agentPrompt,
  });

  return new AgentExecutor({ agent, tools, verbose: true });
}

/**
 * The main function orchestrates the initialization and execution of the langchain logic.
 */
async function runMain() {
  const chatModel = initializeChatModel();
  const { prompt, historyAwarePrompt } = initializePrompts();
  const splitDocs = await fetchContext();
  const retrievalChain = await buildRetrievalChain(chatModel, splitDocs);
  const tools = await setupLangchainTools(chatModel);
  const agentExecutor = await configureAgent(tools, chatModel);

  // Example: Execute retrieval chain with a sample input
  const test = await retrievalChain.invoke({ input: "Who are DHS opponents?" });
  console.log("TEST", test);

  // Example: Execute agent with a sample input
  const agentResult = await agentExecutor.invoke({ input: "how can LangSmith help with testing?" });
  console.log(agentResult.output);
}

// Export `runMain` to be callable from outside this module.
export default runMain;

