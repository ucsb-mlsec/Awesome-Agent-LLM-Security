# Awesome papers and blogs on Agent Security

In the following Google doc., we categorize and summarize recent papers and interesting blogs (about real-world attacks) on the security risks of LLM-enabled AI agents. 
It includes benchmarks, red-teaming (attack), and blue-teaming (defense) approaches 
We use this doc. as a literature review and a paper and blog tracker. 
We also provide discussions on potential research directions.

<a href="https://docs.google.com/document/d/1i5B1tp1srUbPj7nC4mWXRz7OkPbLW3c09r8Qyo6MShU/edit?usp=sharing">AI Agent Safety and Security</a>

It is a commentable link. We welcome new contributors (Feel free to leave your name, we will ack your contribution). 

We also have an old paper list on <a href="https://docs.google.com/document/d/1QowkXo-cM0UQF2FzdSjNkZis9ODeU6AIwucC0nH_vcc/edit?usp=sharing">LLM security and safety</a>, although it is no longer actively maintained. 

# Awesome papers and blogs on Agent Security


## Overview

This repo aims to study the security and safety threats/risks of the LLM-enabled agents. Note that we do not distinguish between safety and security risks in this paper list. The most common distinction would be that security refers to intentional threats, while safety refers to unintentional threats. But sometimes it may not be that easy to distinguish between intentional and unintentional. So we just mix them together. 

**At a high level, having additional components in the system introduces new attack surfaces to the model as well as new attack targets.** Here, the new attack surface to the model mainly refers to indirect prompt injection attacks that attack the models through other components. New attack targets mean non-model components in the system can be attack targets as well. 



1. Attack the model through other components: (Indirect) prompt injection into the model to disrupt the task of the model
    1. Targeted attack: The goal is to hijack the model to accomplish malicious tasks. The tasks can be any model-level threats discussed:
        1. Generic ones from [LLM safety and security](https://docs.google.com/document/d/1QowkXo-cM0UQF2FzdSjNkZis9ODeU6AIwucC0nH_vcc/edit?usp=sharing)
    2. Non-targeted attacks: just disrupt the model to not finish the user/benign task (disrupt instruction following)
2. Attack other components through the model: A system calls and executes other components through the model; attackers can give malicious instructions to the model and let the model generate code that exploits components it interacts with.

For  “Attack the model through other components,” without other components, this attack downgrades to directly generating adversarial prompts to attack target LLMs. This is a common risk across all types of agents. For “Attack other components through the model”, without the model, this attack downgrades to traditional software/system security. This risk category is more complex; the risks are different for different types of agents given their different components and goals. In the following, we first summarize common agent types and then discuss the common risks across all agents (attack the model through other components) and specified risks for each type of agent (Attack other components through the model). 

Based on the attack entry points and attack path/targets, we can have 



1. Entry points, i.e., Injection method:
    1. Direct injection: append the adversarial prompts after **the system prompts**
        1. This is realistic when malicious users try to attack an agent by querying it 
        2. If there is a benign user querying the model and the attacker appends an adversarial prompt after the prompt of the benign user, this IMHO is not that realistic. Indirect injection would be more realistic. 
    2. Indirect injection: inject adversarial prompts through other components in the system 
2. Attack targets and goals:
    3. The model:
        3. Non-target attack: mess up with the instructions following the model
        4. Targeted attack: privacy, harmful content generation, etc.
    4. Other components: case-by-case
        5. E.g., Illusioning and goal-misleading in web agents

	

There could be four combinations, where [direct injection + model as the target] downgrade to [LLM safety and security](https://docs.google.com/document/d/1QowkXo-cM0UQF2FzdSjNkZis9ODeU6AIwucC0nH_vcc/edit?usp=sharing). 

Disclaimer: Please note that while we strive to include a comprehensive selection of papers on the security and safety threats of AI agents, it is not possible to cover all interesting and relevant papers. We will do our best to continually update and expand this document to incorporate more significant contributions in this fast-evolving field. In this document, we discuss the limitations of the included papers. These critiques are not intended to reflect negatively on the authors or the quality of their work. All the papers reviewed here are recognized as valuable contributions to the field. Our aim is to provide constructive analysis to foster further research and understanding.


## Agentic system & benchmarks

A typical agent system is composed of (multiple) LLMs and tools, where LLMs serve as agents to finish a given task, leveraging the available tools. RAG can be considered as one of the simplest agents, where the tool is an external knowledge base. Another simple example of the agent is to connect an LLM with a code interpreter that can execute the code generated by the model. More complex agents involve having multiple collaborative agents, tools, and memory, where each subset of agents controls a different set of tools. Based on their purposes, existing agents can be categorized as web agents that facilitate human-web interactions; coding agents that aid humans in writing code, providing code completion, debugging, etc; and Personal assistant agents that assist users with daily tasks (e.g., setting calendars and sending emails). Note that personal assistant agents can also have web components. Many of the agents involve multiple data modalities.

### Agent survey and benchmarks



1. Ai agents vs. agentic ai: A conceptual taxonomy, applications and challenge
2. From llm reasoning to autonomous ai agents: A comprehensive review
3. From standalone llms to integrated intelligence: A survey of compound al systems
4. A survey of agent interoperability protocols: Model context protocol (mcp), agent communication protocol (acp), agent-to-agent protocol (a2a), and agent network protocol (anp)
5. A Survey of AI Agent Protocols
6. WorkArena: How Capable Are Web Agents at Solving Common Knowledge Work Tasks?
    1. Complete comment tasks related to the web: Dashboard, Form, Knowledge, List-filter, List-sort, Menu, Service Catalog, AXtree, HTML, screenshots
7. Web
    1. *Webarena: A realistic web environment for building autonomous agents*
        1. Shopping, Gitlab, Reddit
    2. *VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks*
        1. Tasks: Classifieds, Shopping, and Reddit
        2. Input: Image, HTML, accessibility tree, SOM(image with numbers)
        3. Propose a new VLM agent, where every task has a visual understanding
        4. Evaluation: exact_match, must_include, fuzzy_match, must_exclude
    4. WebVoyager [https://arxiv.org/pdf/2401.13919](https://arxiv.org/pdf/2401.13919)
    5. WEBLINX: Real-World Website Navigation with Multi-Turn Dialogue
    6. Mind2Web [https://huggingface.co/spaces/osunlp/Online_Mind2Web_Leaderboard](https://huggingface.co/spaces/osunlp/Online_Mind2Web_Leaderboard)
    7. Google Project Mariner [https://deepmind.google/models/project-mariner/](https://deepmind.google/models/project-mariner/)
    8. Browser use [https://browser-use.com/](https://browser-use.com/)
8. *OS/software interaction*
    1. WindowsArena [https://github.com/microsoft/WindowsAgentArena](https://github.com/microsoft/WindowsAgentArena)
        1. Agent design: model+executor (environment executes the model’s output)
        2. Input: UIA tree(Windows UI Automation tree), OCR, Image
        3. Office, Web Browsing, Windows System, Coding, Media, Windows Utilities. 154 tasks
    2. AndroidWorld <span style="text-decoration:underline;"> [https://github.com/google-research/android_world](https://github.com/google-research/android_world)</span>
        1. Screenshot,  ally tree
        2. requires_setup, data_entry, complex_ui_understanding, parameterized, game_playing, multi_app, memorization, math_counting, screen_reading, verification, information_retrieval, transcription,Repetition, search, data_edit
    3. OSWorld[ https://os-world.github.io/](https://os-world.github.io/)
        1. Screenshot+tree
        2. Workflow, Windows-Workflow, Chrome, GIMP, LibreOffice Calc, LibreOffice Impress, LibreOffice Writer, OS, Thunderbird, VLC, VS Code, Excel, Word, PowerPoint
    4. Operating System: ask LLMs to perform os-related tasks. ([https://arxiv.org/abs/2308.03688](https://arxiv.org/abs/2308.03688))
    5. Aios: Llm agent operating system
    6. Android
        1. [https://google-research.github.io/android_world/](https://google-research.github.io/android_world/)
        2. [https://mobilesafetybench.github.io/](https://mobilesafetybench.github.io/)
9. ALFWorld: Daily Household Routines ([https://alfworld.github.io/](https://alfworld.github.io/))

### Agent security survey and benchmarks



1. From Prompt Injections to Protocol Exploits: Threats in LLM-Powered AI Agents Workflows
2. Survey on evaluation of llm-based agents
3. Ai agents under threat: A survey of key security challenges and future pathways
4. Security of AI Agents
5. Securing Agentic AI: A Comprehensive Threat Model and Mitigation Framework for Generative AI Agents
6. [Agentic AI Needs a Systems Theory](https://arxiv.org/pdf/2503.00237 ) (IBM Research)
7. Practices for Governing Agentic AI Systems (OpenAI)
8. Model Context Protocol (MCP): Landscape, Security Threats, and Future Research Directions
9. General indirect prompt injection
    1. INJECAGENT: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents (2024) [[PDF](https://arxiv.org/pdf/2403.02691)]
        1. Threats: Direct harm to users & exfiltration of private data
        2. Prompt generation method: ReAct
    2. [ToolEmu: Identifying the Risks of LM Agents with an LM-Emulated Sandbox](https://openreview.net/pdf?id=GEcwtMk1uA)
        1. A framework that uses an LM to emulate tool execution and enables scalable testing of LM agents against a diverse range of tools and scenarios (benchmark consisting of 36 high-stakes toolkits and 144 test cases)
        2. LM-based automated safety evaluator that examines agent failures and quantifies associated risks
        3. Threat model: user instructions are ambiguous or omit critical details, posing risks when the LM agent fails to properly resolve these ambiguities, simulations
    3. LLMail-Inject: A Dataset from a Realistic Adaptive Prompt Injection Challenge
10. Benchmarks for web agents
    1. WASP: Benchmarking Web Agent Security Against Prompt Injection Attacks (2025)
        1. Prompt injection benchmark based on WebArena
11. Benchmarks for CUA agents
    1. RedTeamCUA: Realistic Adversarial Testing of Computer-Use Agents in Hybrid Web-OS Environments (2025)
        1. CUA agent testing based on OSworld as the backbone (with sandbox and web container) 
        2. A graphic interface between the agent and the environment
    2. ​​Weathering the CUA Storm: Mapping Security Threats in the Rapid Rise of Computer Use Agents
        1. Clickjacking: Domain spoofing (e.g., g00gle.com)
        2. Remote code execution on a sandbox
        3. Chain-of-thought Exposure
        4. Bypassing Human-in-the-loop Safeguards
        5. Indirect prompt injection attacks
        6.  Identity ambiguity and over-delegation
        7. Content harms
    3. Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents (2024)** **
        1. Threat model: The attacker aims to mislead the LLM agent into using a specified tool (third attack type)
        2. 10 scenarios and 10 agents:
            1. IT management, Investment, Legal advice, Medicine, Academic advising, Counseling, E-commerce, Aerospace design, Research, Autonomous  vehicles
        3. 23 Attacks and defenses: 
            1. Direct prompt injection (Naive, Escape, Ignoring, Fake, Combined)
            2. Observation prompt injection (inject adversarial prompts in the external environment) 
            3. Memory poisoning attack (poison long-term memory like RAG database)
            4. PoT backdoor attack (leave a backdoor in the system prompt)
            5. Mixed attack
            6. Defense: Delimiters, Sandwich Prevention, Instructional Prevention, Paraphrasing, Shuffle, LLM-based detection, PPL detection
12. AgentDojo: A Dynamic Environment to Evaluate Attacks and Defenses for LLM Agents (2024) [[PDF](https://arxiv.org/pdf/2406.13352)]
    1. 4 scenarios (agents): Workspace, Slack, Banking, Travel; 97 tasks, 629 security test cases
    2. Basic defense
        1. Format all tool outputs with special delimiters
        2. Delimiters
        3. Prompt injection detection: uses a BERT classifier as a guardrail
        4. Sandwich Prevention: repeats the user instructions after each function call
        5. Tool filter: restricts LLM itself to a set of tools required to solve a given task, before observing any untrusted data
13. Some direct-injection benchmarks 
    1. Formalizing and Benchmarking Prompt Injection Attacks and Defenses (USENIX’24)
        1. Pattern: benign prompts + adversarial prompts
    2. Assessing Prompt Injection Risks in 200+ Custom GPTs 
    3. Tensor Trust: Interpretable Prompt Injection Attacks from an Online Game [ICLR’ 24]
    4. GenTel-Safe: A Unified Benchmark and Shielding Framework for Defending Against Prompt Injection Attacks
    5. CYBERSECEVAL 2: A Wide-Ranging Cybersecurity Evaluation Suite for Large Language Models [[PDF](https://arxiv.org/pdf/2404.13161)]
        1. A section about prompt injection; Two goals: violate application logic (go off-topic)/violate security constraints
    6. A Critical Evaluation of Defenses against Prompt Injection Attacks


### Agent system card



1. Operator system card: [https://openai.com/index/operator-system-card/](https://openai.com/index/operator-system-card/)
2. Lessons from Defending Gemini against Indirect Prompt Injections 


## Red-teaming

### General attacks: Prompt injection/Memory/Backdoor

Note that injection is an attack method, not an attack goal; one can launch an injection attack with different goals



1. **Model backdoor**
    1. Watch Out for Your Agents! Investigating Backdoor Threats to LLM-Based Agents
        1. Insert backdoor triggers into web agents through fine-tuning backbone models with white-box access, aiming to mislead agents into making incorrect purchase decisions
    2. (old)Navigation as attackers wish? towards the building, byzantine-robust embodied agents under federated learning (Data poisoning attack)
    3. BadAgent: Inserting and Activating Backdoor Attacks in LLM Agents (ACL24)
        1. insert backdoor triggers into web agents through fine-tuning backbone models with white-box access
            1. Insert trigger and malicious output into benign data to craft an attack dataset. Then do a classic data-poisoning attack
        2. Threat model: finetune a benign llm with the backdoor dataset. And (1) victims use our released model (2) victims finetune our released model and then use it
2. **Direct prompt injection**: Directly append the malicious prompts into the user prompts 	
    1. UDora: A Unified Red Teaming Framework against LLM Agents by Dynamically Hijacking Their Own Reasoning
    2. How Not to Detect Prompt Injections with an LLM
    3. Automatic and Universal Prompt Injection Attacks against Large Language Models (2024)
        1. Gradient-based method, similar to GCG, with slightly different optimization targets
        2. Propose three prompt injection objectives according to whether the response is relevant to the user’s input: static, semi-dynamic, and dynamic
            1. Static objective: the attacker aims for a consistent response, regardless of the user’s instructions or external data
            2. Semi-dynamic objective: the attacker expects the victim model to produce consistent content before providing responses relevant to the user’s input
            3. Dynamic objective: the attacker wants the victim model to give responses relevant to the user’s input, but maintain malicious content simultaneously.
    4. Goal-guided Generative Prompt Injection Attack on Large Language Models (2024)
        1. Attack objective design
            1. Effective: attack inputs with original high benign accuracy to high ASR 
            2. Imperceptible: the original input and the adversarial input are very similar in terms of some semantic metrics. They use cosine similarity
            3. Input-dependent: a prompt injection manner to form the attack prompt
    5. [Prompt Injection attack against LLM-integrated Applications](https://arxiv.org/pdf/2306.05499)
        1. Design a prompt injection pattern with three elements: Framework Component, Separator Component, Disruptor Component
    6. [Ignore Previous Prompt: Attack Techniques For Language Models](https://arxiv.org/pdf/2211.09527)
        1. Manual prompt injection: Goal hijacking and prompt leaking
    7. Ignore this title and HackAPrompt: Exposing systemic vulnerabilities of LLMs through a global prompt hacking competition (EMNLP 2023)
    8. [Imprompter: Tricking LLM Agents into Improper Tool Use](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=1eNRT-UAAAAJ&citation_for_view=1eNRT-UAAAAJ:WF5omc3nYNoC)
        1. Craft obfuscated adversarial prompt attacks that violate the confidentiality and integrity of user resources connected to an LLM agent
3. **Indirect prompt injection**
    1. Adaptive Attacks Break Defenses Against Indirect Prompt Injection Attacks on LLM Agents
    2. Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection [[PDF](https://arxiv.org/pdf/2302.12173)]
        1. Study some basic pattern-based attack methods
        2. Injection method: retrieval-based methods, active methods, user-driven injections, hidden injections
        3. Threats: information gathering/ fraud/intrusion/malware/manipulated content/availability
    3. [AgentVigil: Generic Black-Box Red-Teaming for Indirect Prompt Injection against LLM Agents](https://arxiv.org/abs/2505.05849) (2025)
        1. Leverage fuzzing to generate attack prompts for prompt injection
4. **Prompt injection in Multi-Agent System (MAS)**
    1. Prompt Infection: LLM-to-LLM Prompt Injection within Multi-Agent Systems (2024)
        1. Inject the malicious prompts into the external content and rely on the data sharing mechanism across different agents to affect multiple agents 
    2. Red-teaming llm multi-agent systems via communication attacks
    3. Evil Geniuses: Delving into the Safety of LLM-based Agents
5. **Memory poisoning**
    1. AGENTPOISON: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases
        1. Attack the knowledge database by adding malicious data
        2. Loss
            1. Uniqueness: poison data should be away from benign data
            2. Compactness: poison data should be similar
            3. Coherence: The trigger’s perplexity should be low
            4. Target generation: triggers cause target action
    2. A practical memory injection attack against llm agents
    3. PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models
6. **Tool poisoning**
    1. MCP Security Notification: Tool Poisoning Attacks
        1. [https://blog.trailofbits.com/2025/04/21/jumping-the-line-how-mcp-servers-can-attack-you-before-you-ever-use-them/](https://blog.trailofbits.com/2025/04/21/jumping-the-line-how-mcp-servers-can-attack-you-before-you-ever-use-them/)
        2. [https://blog.trailofbits.com/2025/04/23/how-mcp-servers-can-steal-your-conversation-history/](https://blog.trailofbits.com/2025/04/23/how-mcp-servers-can-steal-your-conversation-history/)
    2. [Prompt Injection Attack to Tool Selection in LLM Agents](https://arxiv.org/pdf/2504.19793)
7. **Exfiltration attack: Inject URL to exploit renderers that fetches data from attacker’s server, leaking agent data**
    1. EchoLeak cve-2025-32711 [https://www.aim.security/lp/aim-labs-echoleak-blogpost](https://www.aim.security/lp/aim-labs-echoleak-blogpost)
    2. [https://simonwillison.net/tags/exfiltration-attacks/](https://simonwillison.net/tags/exfiltration-attacks/)
8. Some related works that attack the instruction following of LLMs (related to agents but mainly about model)
    1. [An LLM Can Fool Itself: A Prompt-Based Adversarial Attack](https://arxiv.org/pdf/2310.13345)
        1. Audit the LLM’s adversarial robustness via a prompt-based adversarial attack
        2. Let LLMs generate adversarial prompts, and define the generation prompts with three components:
            1. original input (OI), including the original sample and its ground-truth label
            2. attack objective (AO) illustrating a task description of generating a new sample that can fool itself without changing the semantic meaning
            3. attack guidance (AG) containing the perturbation instructions, e.g., add some characters 
    2. [The SIFo Benchmark: Investigating the Sequential Instruction Following Ability of Large Language Models](https://arxiv.org/pdf/2406.19999) (Testing-phase backdoor)
        1. text modification, question answering, mathematics, and *security rule following*
    3. [Can LLMs Follow Simple Rules?](https://arxiv.org/pdf/2311.04235) (Instruction following)
        1. Propose Rule-following Language Evaluation Scenarios (RULES), a programmatic framework for measuring rule-following ability in LLM
        2. Defense: test-time steering and finetuning
    4. [A Trembling House of Cards? Mapping Adversarial Attacks against Language Agents](https://arxiv.org/pdf/2402.10196)
    5. [Misusing Tools in Large Language Models With Visual Adversarial Examples](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=1eNRT-UAAAAJ&citation_for_view=1eNRT-UAAAAJ:zYLM7Y9cAGgC)
        1. Visual input-based prompt injection (applicable to both direct and indirect prompt injections)