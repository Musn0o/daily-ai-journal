### 1- Foundational Large Language Models & Text Generation

**Excellent! Let's break down the "Foundational Large Language Models & Text Generation" whitepaper for you, like you're a complete beginner.**

This whitepaper is all about **Large Language Models (LLMs)**, which are super smart computer programs that can understand and generate text that sounds like it was written by a human [1]. Think of them as really advanced chat-bots or text creators.

**Why are these LLMs important?**

*   They can do many amazing things, like **translate languages**, **write code**, **create stories**, **summarize long articles**, and **answer your questions** [2].
*   They are a big step up from older language models and can perform complex tasks that require understanding and reasoning [2].
*   Even though they are trained on a lot of general information, they can be **adapted to do specific tasks** through a process called **fine-tuning**, which requires less data and effort than training them from scratch [2].
*   We can also guide them to give us the kind of responses we want by using something called **prompt engineering** [2]. This is like giving them clear instructions to get the best results.

**How do LLMs work? The Transformer**

At the heart of most modern LLMs is something called a **Transformer** [3, 4]. Think of the Transformer as the main engine inside these models. It was invented at Google and is really good at understanding relationships between words in a sequence [4].

*   The Transformer takes text as **input** and turns each word into a special kind of number called an **embedding** [5]. This embedding captures the meaning of the word. It also figures out the **position** of each word in the sentence [6].
*   Then, it uses something called **attention** [6]. This allows the model to focus on the most important words in the input when trying to understand the context. Imagine reading a sentence and automatically paying more attention to certain words that tell you what the sentence is about. That's kind of what attention does.
*   The Transformer has two main parts in its original design: an **encoder** and a **decoder** [4].
    *   The **encoder** takes the input text and creates a representation of it that captures the meaning of the entire sentence [7].
    *   The **decoder** then uses this representation to generate the output text, word by word [4, 7]. Many newer LLMs use only the **decoder** part [8].

**Training LLMs**

LLMs learn by being fed massive amounts of text data [1]. This process is called **training** [9].

*   During training, the model tries to **predict the next word** in a sequence [10, 11]. By doing this over and over with tons of text, it learns the patterns and relationships in language.
*   The data needs to be prepared first by cleaning it up and breaking it down into smaller pieces called **tokens** [5, 9].
*   There are different ways to train Transformers, depending on whether they use just a decoder, just an encoder, or both [11, 12].

**Fine-tuning LLMs**

After the initial training, we can further train the LLM on a smaller, more specific dataset to make it better at a particular task. This is called **fine-tuning** [2, 13].

*   **Supervised fine-tuning (SFT)** uses labeled data to teach the model to perform specific tasks, like classifying text or answering questions [13].
*   **Reinforcement learning from human feedback (RLHF)** is another powerful technique where human preferences are used to further refine the model's responses, making them more helpful, truthful, and safe [14].

**Using LLMs: Prompt Engineering**

To get LLMs to do what we want, we need to give them clear instructions, called **prompts** [15, 16].

*   **Prompt engineering** is the art of designing these prompts effectively [2, 15, 17].
*   There are different ways to prompt LLMs, like giving them no examples (**zero-shot**), a few examples (**few-shot**), or guiding their reasoning step by step (**chain-of-thought**) [18, 19].

**Applications of LLMs**

LLMs are being used in many different areas [1, 20]:

*   **Content generation:** Writing articles, stories, and marketing materials [21].
*   **Code and mathematics:** Generating and understanding code, solving math problems [22].
*   **Translation:** Converting text between languages [23].
*   **Text summarization:** Creating shorter versions of long texts [24].
*   **Question answering:** Answering questions based on their knowledge [1].
*   **Text classification:** Categorizing text into different groups [25].
*   **Natural language inference:** Understanding logical relationships between sentences [26].
*   **Multimodal applications:** Processing and generating text along with images, audio, and video [27, 28].

**Key Takeaways**

*   **Transformers** are the foundation of modern LLMs [29].
*   LLMs are trained on huge amounts of data to understand and generate human-like text [1].
*   **Fine-tuning** allows us to adapt LLMs for specific tasks [13].
*   **Prompt engineering** is crucial for guiding LLMs to produce the desired outputs [16].
*   LLMs have a wide range of applications that are constantly expanding [20].

This whitepaper goes into much more detail about the different types of LLM architectures (like GPT, BERT, Gemini), how they are trained, and techniques to make them work faster [30, 31]. But hopefully, this beginner-friendly explanation gives you a good starting point for understanding what these powerful language models are all about!



# 2- Prompt Engineering


This whitepaper, titled "**Prompt Engineering**," is like a guide to learning how to best talk to powerful AI language models to get the answers or text you need. Think of these language models as very intelligent but sometimes not very focused. **Prompt engineering is the art and science of crafting the right instructions, or "prompts," to guide them** [1-4].

Here's a breakdown of what the whitepaper covers for a complete beginner:

*   **What is a Prompt?** A prompt is simply the text you give to a language model as input. It can be a question, the start of a story, or instructions for the model. When you chat with an AI, you're essentially writing prompts [2, 3, 5].

*   **Who Can Do It?** The whitepaper emphasizes that **you don't need to be a data scientist or a machine learning expert to write a prompt; anyone can do it** [2, 6]. However, writing *effective* prompts requires some skill [2, 3].

*   **Why is Prompt Engineering Important?** Just like giving unclear instructions to a person can lead to confusing results, **inadequate prompts can lead to ambiguous or inaccurate responses from language models** [2]. Crafting good prompts helps the model understand what you want and provide meaningful output [2-4]. It's often an iterative process where you try different prompts to see what works best [2, 7].

*   **Controlling the Model's Output (LLM Output Configuration)** [8]: Besides the prompt itself, you can also adjust some settings of the language model to control its output [8]. The whitepaper discusses a few of these:
    *   **Output Length:** You can control how many words or "tokens" the model generates in its response [8]. However, simply reducing the length doesn't make the output more concise; it just stops the model sooner [9]. Output length restriction can be important for techniques like ReAct [9]. Generating more tokens requires more computation and can lead to slower responses and higher costs [8].
    *   **Sampling Controls (Temperature, Top-K, Top-P)** [9]: These settings influence how random or predictable the model's responses are [9].
        *   **Temperature:** A lower temperature (e.g., 0.1 for classification) makes the model more deterministic (it will usually pick the most likely next word), which is good for tasks with a single correct answer [10, 11]. A higher temperature (e.g., 1 for higher creativity) introduces more randomness and creativity [12]. Setting the temperature to 0 makes the model always pick the most probable word [11]. Inappropriate temperature settings can exacerbate the "repetition loop bug" [13, 14].
        *   **Top-K and Top-P:** These are other ways to control which words the model considers when generating its response [13]. They can effectively be disabled by using default values [10]. The best way to choose these settings is often through experimentation [14]. Extreme settings of one can even make the others irrelevant [10]. With more freedom in these settings, the LLM might generate less relevant text [13].

*   **Prompting Techniques** [10]: The main part of the whitepaper explores various techniques to write better prompts [10].

    *   **General Prompting / Zero-Shot:** This is the simplest way, where you just give the model a task description without any examples [10, 15]. For example, you could just ask, "Classify this movie review as positive or negative: 'It was amazing!'". It's called "zero-shot" because you give the model zero examples [5, 15]. An example for movie review classification is provided [16].

    *   **One-Shot and Few-Shot:** When zero-shot doesn't work well enough, you can provide demonstrations or examples in your prompt [17]. A **one-shot prompt** gives the model a single example of what you want [17]. A **few-shot prompt** provides multiple examples (usually 3-5) to show the model a pattern to follow [17]. These examples can help steer the model towards a specific output style or structure [17]. Examples are especially useful when you want to steer the model to a certain output structure or pattern [17]. When choosing examples, they should be relevant, diverse, of high quality, and well-written [18, 19]. Including edge cases in your examples can help generate more robust output [18]. An example of a few-shot prompt for parsing pizza orders to JSON is given [20, 21].

    *   **System, Contextual, and Role Prompting** [18]: These techniques help set the stage for the language model [22]. Distinguishing between these provides a framework for designing prompts with clear intent [22].
        *   **System Prompting:** This sets the overall context and purpose for the model, like telling it it should act as a translator or a classifier [22, 23]. It can also be used to specify how the output should be returned, such as in uppercase [12, 23] or in a specific format like JSON [23-26]. The name ‘system prompt’ stands for ‘providing an additional task to the system’ [23]. System prompts can be useful for safety and toxicity control by adding lines like 'You should be respectful in your answer' [27].
        *   **Role Prompting:** This involves assigning a specific role to the AI model, like a book editor, a teacher, or a travel guide [22, 27]. This helps the model generate more relevant output in the style and tone of that role [27, 28]. Defining a role gives the model a blueprint of the tone, style, and focused expertise you’re looking for [28]. Examples include acting as a travel guide [27-29] with different styles like humorous and inspirational [28, 30].
        *   **Contextual Prompting:** This provides immediate, task-specific information to guide the response [22]. It helps ensure efficient and seamless AI interactions, allowing the model to understand the request quickly and generate accurate responses [30].

    *   **Step-Back Prompting:** This technique encourages the model to first think about a more general question related to your specific task before attempting to answer the specific question [15, 28, 31]. This "step back" can help the model access relevant background knowledge [31]. An example compares a traditional prompt for video game settings with a step-back prompt [31, 32]. Including the answer of the step back as context in the original prompt can influence the output [32].

    *   **Chain of Thought (CoT):** For complex tasks that require reasoning, you can prompt the model to explain its reasoning step-by-step [15, 33]. This can lead to more accurate answers [34], especially for things like math problems [35]. CoT is low-effort, effective with off-the-shelf LLMs, and provides interpretability [33]. It also appears to improve robustness between different LLM versions [33]. However, it leads to more output tokens, increasing cost and time [35]. CoT can be useful for code generation and creating synthetic data [36]. It's based on greedy decoding, so the temperature should be set to 0 [11]. Putting the answer after the reasoning is required [11]. An example demonstrates the flaws of not using CoT for a math problem [35] and then shows a single-shot CoT prompt [36].

    *   **Self-Consistency:** To improve the reliability of answers, especially in reasoning tasks, you can ask the model the same prompt multiple times with a higher temperature setting to encourage diverse reasoning paths [34, 37]. Then, you extract the answer from each response [37] and choose the most common answer [37, 38]. This gives a pseudo-probability likelihood of an answer being correct but has higher costs [37]. An example shows self-consistency for email classification [37-43]. By generating many Chains of Thoughts and taking the most common answer, a more consistently correct result can be obtained [38].

    *   **Tree of Thoughts (ToT):** This is a more advanced technique that allows the model to explore multiple reasoning paths simultaneously, unlike the single path in Chain of Thought [15, 41, 44]. [No further details on ToT are provided in the excerpts.]

    *   **ReAct (Reason & Act):** This technique combines reasoning and taking actions in a loop [15, 42, 44, 45]. The LLM reasons about the problem, generates a plan of action, performs the actions (e.g., using tools like search engines), observes the results, and uses the observations to update its reasoning and plan the next action until a solution is reached [45, 46]. An example shows using LangChain and Vertex AI for a question about Metallica band members' children [46-48]. ReAct requires resending previous prompts/responses and setting up the model with appropriate examples/instructions [48].

    *   **Automatic Prompt Engineering (APE):** This is a method to automatically generate and evaluate different prompts for a specific task, reducing the need for manual effort [44, 45, 49]. You prompt a model to generate more prompts, evaluate them (e.g., using BLEU or ROUGE metrics) [49, 50], and select the best one [49, 51]. This can be helpful for tasks like training a chatbot for a webshop [49, 52]. An example shows generating variants for ordering a t-shirt [50, 52].

    *   **Code Prompting** [44, 46]: The whitepaper also discusses how to use prompts for various coding tasks [51]. Gemini primarily focuses on text-based prompts, including those for code [51].
        *   **Prompts for writing code:** Gemini can help with writing code in any programming language [51]. An example shows a prompt to write a Bash script for renaming files in a folder [53-55]. It's essential to read and test the generated code [55, 56].
        *   **Prompts for explaining code:** LLMs can help understand someone else's code [57]. An example shows a prompt to explain a Bash code snippet [57, 58].
        *   **Prompts for translating code:** LLMs can assist in converting code from one programming language to another [59]. An example demonstrates translating Bash code to Python [59-62]. Remember to check for proper indenting when prompting for Python code in Vertex AI Studio [62].
        *   **Prompts for debugging and reviewing code:** LLMs can help identify errors and suggest improvements in code [63, 64]. An example shows debugging a broken Python script and getting suggestions for improvement [64-70].

    *   **Multimodal Prompting:** This briefly mentions using different types of inputs beyond just text, such as images, to guide the model [44, 59, 71], but it notes that this is a separate topic [71].

*   **Best Practices** [19, 59, 72]: The whitepaper concludes with several best practices for effective prompt engineering [19].
    *   **Provide examples:** Using one-shot or few-shot examples is highly effective [19, 59].
    *   **Design with simplicity:** Keep prompts concise, clear, and easy to understand. Use action verbs [60, 73, 74].
    *   **Be specific about the output:** Provide specific details about the desired format, style, or content [25, 61].
    *   **Use Instructions over Constraints:** Focus on positive instructions rather than what to avoid, unless necessary for safety or clarity [61, 75, 76].
    *   **Control the max token length:** Set a limit in the configuration or explicitly request a length in the prompt [63, 77].
    *   **Use variables in prompts:** Employ variables for reuse and dynamic prompts [29, 63, 77].
    *   **Experiment with input formats and writing styles:** Try different phrasings (question, statement, instruction) [29, 78, 79].
    *   **For few-shot prompting with classification tasks, mix up the classes** in the examples [78, 80].
    *   **Adapt to model updates:** Stay informed about model changes and adjust prompts accordingly [64, 80].
    *   **Experiment with output formats:** For non-creative tasks, request structured formats like JSON or XML [26, 64, 80].
    *   **JSON Repair:** Use tools to fix incomplete JSON output [65, 81].
    *   **Working with Schemas:** Structure your input data using JSON Schemas to provide a clear blueprint to the LLM [66, 82-84].
    *   **Experiment together with other prompt engineers** to get diverse ideas [67, 85].
    *   **CoT Best Practices:** Put the answer after reasoning and set the temperature to 0 [11, 68]. Extract the final answer from the reasoning [11].
    *   **Document the various prompt attempts:** Keep detailed records of your prompts, configurations, and outputs to learn and debug [68, 86-88]. A template for documentation is provided [7, 87, 89]. Save prompts separately from code in operationalized systems and use automated tests [88].

In summary, the "**Prompt Engineering**" whitepaper provides a foundational understanding of how to interact effectively with large language models by crafting well-designed prompts and adjusting output configurations. It introduces various techniques and best practices to help you get the desired results for a wide range of tasks [70, 89, 90].



