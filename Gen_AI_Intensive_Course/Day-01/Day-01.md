### 1- Foundational Large Language Models & Text Generation

**Excellent! Let's break down the "Foundational Large Language Models & Text Generation" whitepaper for you, like you're a complete beginner.**

This whitepaper is all about **Large Language Models (LLMs)**, which are super smart computer programs that can understand and generate text that sounds like it was written by a human. Think of them as really advanced chat-bots or text creators.

**Why are these LLMs important?**

*   They can do many amazing things, like **translate languages**, **write code**, **create stories**, **summarize long articles**, and **answer your questions** .
*   They are a big step up from older language models and can perform complex tasks that require understanding and reasoning .
*   Even though they are trained on a lot of general information, they can be **adapted to do specific tasks** through a process called **fine-tuning**, which requires less data and effort than training them from scratch e.g. A general "Engineer" has a broad understanding of engineering principles. When you "fine-tune" them by having them specialize in a specific area like civil engineering, and then even further into, say, bridge design or structural analysis, they become much more expert and effective in that particular niche.
*   We can also guide them to give us the kind of responses we want by using something called **prompt engineering** . This is like giving them clear instructions to get the best results.

**How do LLMs work? The Transformer**

At the heart of most modern LLMs is something called a **Transformer** . Think of the Transformer as the main engine inside these models. It was invented at Google and is really good at understanding relationships between words in a sequence .

*   The Transformer takes text as **input** and turns each word into a special kind of number called an **embedding** . This embedding captures the meaning of the word. It also figures out the **position** of each word in the sentence .
*   Then, it uses something called **attention** . This allows the model to focus on the most important words in the input when trying to understand the context. Imagine reading a sentence and automatically paying more attention to certain words that tell you what the sentence is about. That's kind of what attention does.
*   The Transformer has two main parts in its original design: an **encoder** and a **decoder** .
    *   The **encoder** takes the input text and creates a representation of it that captures the meaning of the entire sentence .
    *   The **decoder** then uses this representation to generate the output text, word by word . Many newer LLMs use only the **decoder** part .

**Training LLMs**

LLMs learn by being fed massive amounts of text data . This process is called **training** .

*   During training, the model tries to **predict the next word** in a sequence . By doing this over and over with tons of text, it learns the patterns and relationships in language.
*   The data needs to be prepared first by cleaning it up and breaking it down into smaller pieces called **tokens** .
*   There are different ways to train Transformers, depending on whether they use just a decoder, just an encoder, or both .

**Fine-tuning LLMs**

After the initial training, we can further train the LLM on a smaller, more specific dataset to make it better at a particular task. This is called **fine-tuning** .

*   **Supervised fine-tuning (SFT)** uses labeled data to teach the model to perform specific tasks, like classifying text or answering questions .
*   **Reinforcement learning from human feedback (RLHF)** is another powerful technique where human preferences are used to further refine the model's responses, making them more helpful, truthful, and safe .

**Using LLMs: Prompt Engineering**

To get LLMs to do what we want, we need to give them clear instructions, called **prompts** .

*   **Prompt engineering** is the art of designing these prompts effectively .
*   There are different ways to prompt LLMs, like giving them no examples (**zero-shot**), a few examples (**few-shot**), or guiding their reasoning step by step (**chain-of-thought**) .

**Applications of LLMs**

LLMs are being used in many different areas :

*   **Content generation:** Writing articles, stories, and marketing materials .
*   **Code and mathematics:** Generating and understanding code, solving math problems .
*   **Translation:** Converting text between languages .
*   **Text summarization:** Creating shorter versions of long texts .
*   **Question answering:** Answering questions based on their knowledge .
*   **Text classification:** Categorizing text into different groups .
*   **Natural language inference:** Understanding logical relationships between sentences .
*   **Multimodal applications:** Processing and generating text along with images, audio, and video .

**Key Takeaways**

*   **Transformers** are the foundation of modern LLMs .
*   LLMs are trained on huge amounts of data to understand and generate human-like text .
*   **Fine-tuning** allows us to adapt LLMs for specific tasks .
*   **Prompt engineering** is crucial for guiding LLMs to produce the desired outputs .
*   LLMs have a wide range of applications that are constantly expanding .

This whitepaper goes into much more detail about the different types of LLM architectures (like GPT, BERT, Gemini), how they are trained, and techniques to make them work faster . But hopefully, this beginner-friendly explanation gives you a good starting point for understanding what these powerful language models are all about!



# 2- Prompt Engineering


This whitepaper, titled "**Prompt Engineering**," is like a guide to learning how to best talk to powerful AI language models to get the answers or text you need. Think of these language models as very intelligent but sometimes not very focused. **Prompt engineering is the art and science of crafting the right instructions, or "prompts," to guide them** .

Here's a breakdown of what the whitepaper covers for a complete beginner:

*   **What is a Prompt?** A prompt is simply the text you give to a language model as input. It can be a question, the start of a story, or instructions for the model. When you chat with an AI, you're essentially writing prompts .

*   **Who Can Do It?** The whitepaper emphasizes that **you don't need to be a data scientist or a machine learning expert to write a prompt; anyone can do it** . However, writing *effective* prompts requires some skill .

*   **Why is Prompt Engineering Important?** Just like giving unclear instructions to a person can lead to confusing results, **inadequate prompts can lead to ambiguous or inaccurate responses from language models** . Crafting good prompts helps the model understand what you want and provide meaningful output . It's often an iterative process where you try different prompts to see what works best .

*   **Controlling the Model's Output (LLM Output Configuration)** : Besides the prompt itself, you can also adjust some settings of the language model to control its output . The whitepaper discusses a few of these:
    *   **Output Length:** You can control how many words or "tokens" the model generates in its response . However, simply reducing the length doesn't make the output more concise; it just stops the model sooner . Output length restriction can be important for techniques like ReAct . Generating more tokens requires more computation and can lead to slower responses and higher costs .
    *   **Sampling Controls (Temperature, Top-K, Top-P)** : These settings influence how random or predictable the model's responses are .
        *   **Temperature:** A lower temperature (e.g., 0.1 for classification) makes the model more deterministic (it will usually pick the most likely next word), which is good for tasks with a single correct answer . A higher temperature (e.g., 1 for higher creativity) introduces more randomness and creativity . Setting the temperature to 0 makes the model always pick the most probable word . Inappropriate temperature settings can exacerbate the "repetition loop bug" .
        *   **Top-K and Top-P:** These are other ways to control which words the model considers when generating its response . They can effectively be disabled by using default values . The best way to choose these settings is often through experimentation . Extreme settings of one can even make the others irrelevant . With more freedom in these settings, the LLM might generate less relevant text .

*   **Prompting Techniques** : The main part of the whitepaper explores various techniques to write better prompts .

    *   **General Prompting / Zero-Shot:** This is the simplest way, where you just give the model a task description without any examples . For example, you could just ask, "Classify this movie review as positive or negative: 'It was amazing!'". It's called "zero-shot" because you give the model zero examples. An example for movie review classification is provided.

    *   **One-Shot and Few-Shot:** When zero-shot doesn't work well enough, you can provide demonstrations or examples in your prompt . A **one-shot prompt** gives the model a single example of what you want . A **few-shot prompt** provides multiple examples (usually 3-5) to show the model a pattern to follow . These examples can help steer the model towards a specific output style or structure . Examples are especially useful when you want to steer the model to a certain output structure or pattern . When choosing examples, they should be relevant, diverse, of high quality, and well-written . Including edge cases in your examples can help generate more robust output . An example of a few-shot prompt for parsing pizza orders to JSON is given.

    *   **System, Contextual, and Role Prompting** : These techniques help set the stage for the language model . Distinguishing between these provides a framework for designing prompts with clear intent .
        *   **System Prompting:** This sets the overall context and purpose for the model, like telling it it should act as a translator or a classifier . It can also be used to specify how the output should be returned, such as in uppercase  or in a specific format like JSON . The name ‘system prompt’ stands for ‘providing an additional task to the system’ . System prompts can be useful for safety and toxicity control by adding lines like 'You should be respectful in your answer' .
        *   **Role Prompting:** This involves assigning a specific role to the AI model, like a book editor, a teacher, or a travel guide. This helps the model generate more relevant output in the style and tone of that role . Defining a role gives the model a blueprint of the tone, style, and focused expertise you’re looking for . Examples include acting as a travel guide  with different styles like humorous and inspirational .
        *   **Contextual Prompting:** This provides immediate, task-specific information to guide the response . It helps ensure efficient and seamless AI interactions, allowing the model to understand the request quickly and generate accurate responses .

    *   **Step-Back Prompting:** This technique encourages the model to first think about a more general question related to your specific task before attempting to answer the specific question . This "step back" can help the model access relevant background knowledge . An example compares a traditional prompt for video game settings with a step-back prompt . Including the answer of the step back as context in the original prompt can influence the output .

    *   **Chain of Thought (CoT):** For complex tasks that require reasoning, you can prompt the model to explain its reasoning step-by-step . This can lead to more accurate answers , especially for things like math problems . CoT is low-effort, effective with off-the-shelf LLMs, and provides interpretability . It also appears to improve robustness between different LLM versions . However, it leads to more output tokens, increasing cost and time . CoT can be useful for code generation and creating synthetic data . It's based on greedy decoding, so the temperature should be set to 0 . Putting the answer after the reasoning is required . An example demonstrates the flaws of not using CoT for a math problem  and then shows a single-shot CoT prompt .

    *   **Self-Consistency:** To improve the reliability of answers, especially in reasoning tasks, you can ask the model the same prompt multiple times with a higher temperature setting to encourage diverse reasoning paths . Then, you extract the answer from each response  and choose the most common answer. This gives a pseudo-probability likelihood of an answer being correct but has higher costs . An example shows self-consistency for email classification . By generating many Chains of Thoughts and taking the most common answer, a more consistently correct result can be obtained .

    *   **Tree of Thoughts (ToT):** This is a more advanced technique that allows the model to explore multiple reasoning paths simultaneously, unlike the single path in Chain of Thought . No further details on ToT are provided in the excerpts.

    *   **ReAct (Reason & Act):** This technique combines reasoning and taking actions in a loop . The LLM reasons about the problem, generates a plan of action, performs the actions (e.g., using tools like search engines), observes the results, and uses the observations to update its reasoning and plan the next action until a solution is reached . An example shows using LangChain and Vertex AI for a question about Metallica band members' children . ReAct requires resending previous prompts/responses and setting up the model with appropriate examples/instructions .

    *   **Automatic Prompt Engineering (APE):** This is a method to automatically generate and evaluate different prompts for a specific task, reducing the need for manual effort. You prompt a model to generate more prompts, evaluate them (e.g., using BLEU or ROUGE metrics), and select the best one . This can be helpful for tasks like training a chatbot for a webshop . An example shows generating variants for ordering a t-shirt .

    *   **Code Prompting** : The whitepaper also discusses how to use prompts for various coding tasks . Gemini primarily focuses on text-based prompts, including those for code .
        *   **Prompts for writing code:** Gemini can help with writing code in any programming language . An example shows a prompt to write a Bash script for renaming files in a folder . It's essential to read and test the generated code .
        *   **Prompts for explaining code:** LLMs can help understand someone else's code . An example shows a prompt to explain a Bash code snippet .
        *   **Prompts for translating code:** LLMs can assist in converting code from one programming language to another . An example demonstrates translating Bash code to Python . Remember to check for proper indenting when prompting for Python code in Vertex AI Studio .
        *   **Prompts for debugging and reviewing code:** LLMs can help identify errors and suggest improvements in code. An example shows debugging a broken Python script and getting suggestions for improvement .

    *   **Multimodal Prompting:** This briefly mentions using different types of inputs beyond just text, such as images, to guide the model , but it notes that this is a separate topic .

*   **Best Practices**: The whitepaper concludes with several best practices for effective prompt engineering .
    *   **Provide examples:** Using one-shot or few-shot examples is highly effective.
    *   **Design with simplicity:** Keep prompts concise, clear, and easy to understand. Use action verbs .
    *   **Be specific about the output:** Provide specific details about the desired format, style, or content .
    *   **Use Instructions over Constraints:** Focus on positive instructions rather than what to avoid, unless necessary for safety or clarity .
    *   **Control the max token length:** Set a limit in the configuration or explicitly request a length in the prompt .
    *   **Use variables in prompts:** Employ variables for reuse and dynamic prompts.
    *   **Experiment with input formats and writing styles:** Try different phrasings (question, statement, instruction) .
    *   **For few-shot prompting with classification tasks, mix up the classes** in the examples .
    *   **Adapt to model updates:** Stay informed about model changes and adjust prompts accordingly .
    *   **Experiment with output formats:** For non-creative tasks, request structured formats like JSON or XML .
    *   **JSON Repair:** Use tools to fix incomplete JSON output .
    *   **Working with Schemas:** Structure your input data using JSON Schemas to provide a clear blueprint to the LLM .
    *   **Experiment together with other prompt engineers** to get diverse ideas .
    *   **CoT Best Practices:** Put the answer after reasoning and set the temperature to 0 . Extract the final answer from the reasoning .
    *   **Document the various prompt attempts:** Keep detailed records of your prompts, configurations, and outputs to learn and debug . A template for documentation is provided. Save prompts separately from code in operationalized systems and use automated tests.

In summary, the "**Prompt Engineering**" whitepaper provides a foundational understanding of how to interact effectively with large language models by crafting well-designed prompts and adjusting output configurations. It introduces various techniques and best practices to help you get the desired results for a wide range of tasks .



