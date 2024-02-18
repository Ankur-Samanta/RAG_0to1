- Generation:
    - Call a language model with a prompt template to generate answers to the user questions over the ingested knowledge base.

At its core, the LLM class utilizes a pre-trained language model, configured through the MistralClient, to process user prompts and generate contextually relevant responses. The class is designed to handle different types of interactions:

RAG Chat: Tailored for queries requiring information retrieval from external sources, it constructs messages that guide the model to reference and synthesize information, enhancing the relevance and accuracy of the response.
Standard Chat: Facilitates general chat interactions without the need for external information, focusing on engaging and contextually appropriate dialogue.
Intent Classification: Analyzes the user's query to determine its intent, categorizing it into predefined classes. This function is crucial for deciding whether a query should trigger a RAG response or a simpler chat interaction, optimizing system resource use and response time.
Streaming: Supports continuous interaction, allowing for a dynamic and responsive chat experience that adapts to the user's ongoing input.
The construct_message_* methods craft messages with specific roles and contents, guiding the language model's response generation process. For example, in RAG interactions, the system message informs the model of its role as a knowledgeable assistant capable of referencing external information, thereby setting the stage for informed and accurate responses.