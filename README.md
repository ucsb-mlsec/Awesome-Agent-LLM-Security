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


## Table of Contents

- [Agentic system & benchmarks](#agentic-system--benchmarks)
  - [Agent survey and benchmarks](#agent-survey-and-benchmarks)
  - [Agent security survey and benchmarks](#agent-security-survey-and-benchmarks)
  - [Agent system card](#agent-system-card)
- [Red-teaming](#red-teaming)
  - [General attacks: Prompt injection/Memory/Backdoor](#general-attacks-prompt-injectionmemorybackdoor)
  - [Attack against specific agents](#attack-against-specific-agents)
- [Blue-teaming](#blue-teaming)
  - [Model-based defenses](#model-based-defenses)
  - [System-level Runtime Defense](#system-level-runtime-defense)
  - [Others](#others)
- [Contributors](#contributors)


## Agentic system & benchmarks

A typical agent system is composed of (multiple) LLMs and tools, where LLMs serve as agents to finish a given task, leveraging the available tools. RAG can be considered as one of the simplest agents, where the tool is an external knowledge base. Another simple example of the agent is to connect an LLM with a code interpreter that can execute the code generated by the model. More complex agents involve having multiple collaborative agents, tools, and memory, where each subset of agents controls a different set of tools. Based on their purposes, existing agents can be categorized as web agents that facilitate human-web interactions; coding agents that aid humans in writing code, providing code completion, debugging, etc; and Personal assistant agents that assist users with daily tasks (e.g., setting calendars and sending emails). Note that personal assistant agents can also have web components. Many of the agents involve multiple data modalities.

### Agent survey and benchmarks



1. Ai agents vs. agentic ai: A conceptual taxonomy, applications and challenge [[Information Fusion'25/05](https://arxiv.org/pdf/2505.10468)]
2. From llm reasoning to autonomous ai agents: A comprehensive review [[arxiv'25/04](https://arxiv.org/abs/2504.19678)]
3. From standalone llms to integrated intelligence: A survey of compound al systems [[arxiv'25/06](https://arxiv.org/abs/2506.04565)]
4. A survey of agent interoperability protocols: Model context protocol (mcp), agent communication protocol (acp), agent-to-agent protocol (a2a), and agent network protocol (anp) [[arxiv'25/04](https://arxiv.org/abs/2505.02279)]
5. A Survey of AI Agent Protocols [[arviv'25/04](A Survey of AI Agent Protocols)]
6. WorkArena: How Capable Are Web Agents at Solving Common Knowledge Work Tasks? [[ICML'24/07](https://servicenow.github.io/WorkArena/)]
    1. Complete comment tasks related to the web: Dashboard, Form, Knowledge, List-filter, List-sort, Menu, Service Catalog, AXtree, HTML, screenshots
7. Web
    1. *Webarena: A realistic web environment for building autonomous agents* [[ICLR'24/05](https://arxiv.org/pdf/2307.13854)]
        1. Shopping, Gitlab, Reddit
    2. *VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks* [[ACL'24/08](https://jykoh.com/vwa)]
        1. Tasks: Classifieds, Shopping, and Reddit
        2. Input: Image, HTML, accessibility tree, SOM(image with numbers)
        3. Propose a new VLM agent, where every task has a visual understanding
        4. Evaluation: exact_match, must_include, fuzzy_match, must_exclude
    4. WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models [[ACL'24/08](https://arxiv.org/pdf/2401.13919)]
    5. WEBLINX: Real-World Website Navigation with Multi-Turn Dialogue [[ICML'24/07](https://arxiv.org/abs/2402.05930)]
    6. An Illusion of Progress? Assessing the Current State of Web Agents [[COLM'25/10](https://huggingface.co/spaces/osunlp/Online_Mind2Web_Leaderboard)]
    7. Google Project Mariner [[https://deepmind.google/models/project-mariner/](https://deepmind.google/models/project-mariner/)]
    8. Browser use [[https://browser-use.com/](https://browser-use.com/)]
8. *OS/software interaction*
    1. Windows Agent Arena: Evaluating Multi-Modal OS Agents at Scale [[arxiv'24/09](https://github.com/microsoft/WindowsAgentArena)]
        1. Agent design: model+executor (environment executes the model’s output)
        2. Input: UIA tree(Windows UI Automation tree), OCR, Image
        3. Office, Web Browsing, Windows System, Coding, Media, Windows Utilities. 154 tasks
    2. AndroidWorld: A Dynamic Benchmarking Environment for Autonomous Agents [[ICLR'25/04](https://github.com/google-research/android_world)]
        1. Screenshot,  ally tree
        2. requires_setup, data_entry, complex_ui_understanding, parameterized, game_playing, multi_app, memorization, math_counting, screen_reading, verification, information_retrieval, transcription,Repetition, search, data_edit
    3. OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments [[NeurIPS'24/12](https://os-world.github.io/)]
        1. Screenshot+tree
        2. Workflow, Windows-Workflow, Chrome, GIMP, LibreOffice Calc, LibreOffice Impress, LibreOffice Writer, OS, Thunderbird, VLC, VS Code, Excel, Word, PowerPoint
    4. AgentBench: Evaluating LLMs as Agents [[ICLR'24/05](https://arxiv.org/abs/2308.03688)]
    5. AIOS: LLM Agent Operating System [[COLM'25/10](https://arxiv.org/abs/2403.16971)]
    6. Android
        1. AndroidWorld: A Dynamic Benchmarking Environment for Autonomous Agents [[ICLR'25/04](https://github.com/google-research/android_world)]
        2. MobileSafetyBench: Evaluating Safety of Autonomous Agents in Mobile Device Control [[arxiv'24/12](https://mobilesafetybench.github.io/)]
9. ALFWorld: Aligning Text and Embodied Environments for Interactive Learning [[ICLR'21/05](https://alfworld.github.io/)]

### Agent security survey and benchmarks



1. From Prompt Injections to Protocol Exploits: Threats in LLM-Powered AI Agents Workflows [[arxiv'25/06](https://arxiv.org/abs/2506.23260)]
2. Survey on evaluation of llm-based agents [[arxiv'25/05](https://arxiv.org/abs/2503.16416)]
3. Ai agents under threat: A survey of key security challenges and future pathways [[ACM Computing Surveys'25/02](https://dl.acm.org/doi/10.1145/3716628)]
4. Security of AI Agents [[arxiv'24/12](https://arxiv.org/abs/2406.08689)]
5. Securing Agentic AI: A Comprehensive Threat Model and Mitigation Framework for Generative AI Agents [[arxiv'25/05](https://arxiv.org/abs/2504.19956)]
6. Agentic AI Needs a Systems Theory (IBM Research) [[arxiv'25/02](https://arxiv.org/pdf/2503.00237)]
7. Practices for Governing Agentic AI Systems (OpenAI) [[openai paper'23'12](https://openai.com/index/practices-for-governing-agentic-ai-systems/)]
8. Model Context Protocol (MCP): Landscape, Security Threats, and Future Research Directions [[arxiv'25/04](https://arxiv.org/abs/2503.23278)]
9. General indirect prompt injection
    1. INJECAGENT: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents [[ACL'24/08](https://arxiv.org/pdf/2403.02691)]
        1. Threats: Direct harm to users & exfiltration of private data
        2. Prompt generation method: ReAct
    2. Identifying the Risks of LM Agents with an LM-Emulated Sandbox [[ICLR'24/05](https://arxiv.org/abs/2309.15817)]
        1. A framework that uses an LM to emulate tool execution and enables scalable testing of LM agents against a diverse range of tools and scenarios (benchmark consisting of 36 high-stakes toolkits and 144 test cases)
        2. LM-based automated safety evaluator that examines agent failures and quantifies associated risks
        3. Threat model: user instructions are ambiguous or omit critical details, posing risks when the LM agent fails to properly resolve these ambiguities, simulations
    3. LLMail-Inject: A Dataset from a Realistic Adaptive Prompt Injection Challenge [[arxiv'25/06](https://arxiv.org/abs/2506.09956)]
10. Benchmarks for web agents
    1. WASP: Benchmarking Web Agent Security Against Prompt Injection Attacks [[arxiv'25/04](https://arxiv.org/abs/2504.18575)]
        1. Prompt injection benchmark based on WebArena
11. Benchmarks for CUA agents
    1. RedTeamCUA: Realistic Adversarial Testing of Computer-Use Agents in Hybrid Web-OS Environments [[arxiv'25/06](https://arxiv.org/abs/2505.21936)]
        1. CUA agent testing based on OSworld as the backbone (with sandbox and web container) 
        2. A graphic interface between the agent and the environment
    2. ​​Weathering the CUA Storm: Mapping Security Threats in the Rapid Rise of Computer Use Agents [[ICML Workshop on Computer Use Agents'25/07](https://openreview.net/pdf/74675450995f897873e67dd1d69351d8b3b3cd38.pdf)]
        1. Clickjacking: Domain spoofing (e.g., g00gle.com)
        2. Remote code execution on a sandbox
        3. Chain-of-thought Exposure
        4. Bypassing Human-in-the-loop Safeguards
        5. Indirect prompt injection attacks
        6.  Identity ambiguity and over-delegation
        7. Content harms
    3. Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents [[ICLR'25/07](https://arxiv.org/abs/2410.02644)]
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
12. AgentDojo: A Dynamic Environment to Evaluate Attacks and Defenses for LLM Agents (2024) [[NeurIPS'24/12](https://arxiv.org/pdf/2406.13352)]
    1. 4 scenarios (agents): Workspace, Slack, Banking, Travel; 97 tasks, 629 security test cases
    2. Basic defense
        1. Format all tool outputs with special delimiters
        2. Delimiters
        3. Prompt injection detection: uses a BERT classifier as a guardrail
        4. Sandwich Prevention: repeats the user instructions after each function call
        5. Tool filter: restricts LLM itself to a set of tools required to solve a given task, before observing any untrusted data
13. Some direct-injection benchmarks 
    1. Formalizing and Benchmarking Prompt Injection Attacks and Defenses [[USENIX'24/08](https://arxiv.org/abs/2310.12815)]
        1. Pattern: benign prompts + adversarial prompts
    2. Assessing Prompt Injection Risks in 200+ Custom GPTs [[ICLR Workshop on Secure and Trustworthy Large Language Models'24/05](https://arxiv.org/pdf/2311.11538)]
    3. Tensor Trust: Interpretable Prompt Injection Attacks from an Online Game [[ICLR'2024/05](https://arxiv.org/pdf/2311.01011)]
    4. GenTel-Safe: A Unified Benchmark and Shielding Framework for Defending Against Prompt Injection Attacks [[arxiv'24/09](https://gentellab.github.io/gentel-safe.github.io/)]
    5. CYBERSECEVAL 2: A Wide-Ranging Cybersecurity Evaluation Suite for Large Language Models [[arxiv'24/04](https://arxiv.org/pdf/2404.13161)]
        1. A section about prompt injection; Two goals: violate application logic (go off-topic)/violate security constraints
    6. A Critical Evaluation of Defenses against Prompt Injection Attacks [[arxiv'25/05](https://arxiv.org/abs/2505.18333)]


### Agent system card



1. Operator system card [[openai blog'25/01](https://openai.com/index/operator-system-card/)]
2. Lessons from Defending Gemini against Indirect Prompt Injections (Google Deepmind) [[arxiv'25/05](https://arxiv.org/abs/2505.14534)]


## Red-teaming

### General attacks: Prompt injection/Memory/Backdoor

Note that injection is an attack method, not an attack goal; one can launch an injection attack with different goals


1. **Model backdoor**
    1. Watch Out for Your Agents! Investigating Backdoor Threats to LLM-Based Agents [[NeurIPS'24/12](https://arxiv.org/abs/2402.11208)]
        1. Insert backdoor triggers into web agents through fine-tuning backbone models with white-box access, aiming to mislead agents into making incorrect purchase decisions
    2. Navigation as attackers wish? towards the building, byzantine-robust embodied agents under federated learning (Data poisoning attack) [[NAACL'24/06](https://arxiv.org/abs/2211.14769)]
    3. BadAgent: Inserting and Activating Backdoor Attacks in LLM Agents [ACL'24/08](https://arxiv.org/abs/2406.03007)
        1. insert backdoor triggers into web agents through fine-tuning backbone models with white-box access
            1. Insert trigger and malicious output into benign data to craft an attack dataset. Then do a classic data-poisoning attack
        2. Threat model: finetune a benign llm with the backdoor dataset. And (1) victims use our released model (2) victims finetune our released model and then use it
2. **Direct prompt injection**: Directly append the malicious prompts into the user prompts 	
    1. UDora: A Unified Red Teaming Framework against LLM Agents by Dynamically Hijacking Their Own Reasoning [[ICML'25/07](https://arxiv.org/abs/2503.01908)]
    2. How Not to Detect Prompt Injections with an LLM [[arxiv'25/07](https://arxiv.org/abs/2507.05630)]
    3. Automatic and Universal Prompt Injection Attacks against Large Language Models [[arxiv'24/05](https://arxiv.org/abs/2403.04957)]
        1. Gradient-based method, similar to GCG, with slightly different optimization targets
        2. Propose three prompt injection objectives according to whether the response is relevant to the user’s input: static, semi-dynamic, and dynamic
            1. Static objective: the attacker aims for a consistent response, regardless of the user’s instructions or external data
            2. Semi-dynamic objective: the attacker expects the victim model to produce consistent content before providing responses relevant to the user’s input
            3. Dynamic objective: the attacker wants the victim model to give responses relevant to the user’s input, but maintain malicious content simultaneously.
    4. Goal-guided Generative Prompt Injection Attack on Large Language Models [[arxiv'24/09](https://arxiv.org/abs/2404.07234)]
        1. Attack objective design
            1. Effective: attack inputs with original high benign accuracy to high ASR 
            2. Imperceptible: the original input and the adversarial input are very similar in terms of some semantic metrics. They use cosine similarity
            3. Input-dependent: a prompt injection manner to form the attack prompt
    5. Prompt Injection attack against LLM-integrated Applications [[arxiv'24/03](https://arxiv.org/pdf/2306.05499)]
        1. Design a prompt injection pattern with three elements: Framework Component, Separator Component, Disruptor Component
    6. Ignore Previous Prompt: Attack Techniques For Language Models [NeurIPS Workshop of ML Safety'22/09](https://arxiv.org/pdf/2211.09527)
        1. Manual prompt injection: Goal hijacking and prompt leaking
    7. Ignore this title and HackAPrompt: Exposing systemic vulnerabilities of LLMs through a global prompt hacking competition [[EMNLP'23/12](https://arxiv.org/abs/2311.16119)]
    8. Imprompter: Tricking LLM Agents into Improper Tool Use [[arxiv'24/10](https://arxiv.org/abs/2410.14923)]
        1. Craft obfuscated adversarial prompt attacks that violate the confidentiality and integrity of user resources connected to an LLM agent
3. **Indirect prompt injection**
    1. Adaptive Attacks Break Defenses Against Indirect Prompt Injection Attacks on LLM Agents [[NAACL'25/04](https://arxiv.org/abs/2503.00061)]
    2. Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection [[ACM CCS Workshop of AISec'23/11](https://arxiv.org/pdf/2302.12173)]
        1. Study some basic pattern-based attack methods
        2. Injection method: retrieval-based methods, active methods, user-driven injections, hidden injections
        3. Threats: information gathering/ fraud/intrusion/malware/manipulated content/availability
    3. AgentVigil: Generic Black-Box Red-Teaming for Indirect Prompt Injection against LLM Agents [[EMNLP'25/11](https://arxiv.org/abs/2505.05849)]
        1. Leverage fuzzing to generate attack prompts for prompt injection
4. **Prompt injection in Multi-Agent System (MAS)**
    1. Prompt Infection: LLM-to-LLM Prompt Injection within Multi-Agent Systems [[arxiv'24/10](https://arxiv.org/abs/2410.07283)]
        1. Inject the malicious prompts into the external content and rely on the data sharing mechanism across different agents to affect multiple agents 
    2. Red-teaming llm multi-agent systems via communication attacks [[ACL'25/07](https://arxiv.org/abs/2502.14847)]
    3. Evil Geniuses: Delving into the Safety of LLM-based Agents [[arxiv'24/02](https://arxiv.org/abs/2311.11855)]
5. **Memory poisoning**
    1. AGENTPOISON: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases [[NeurIPS'24/12](https://arxiv.org/abs/2407.12784)]
        1. Attack the knowledge database by adding malicious data
        2. Loss
            1. Uniqueness: poison data should be away from benign data
            2. Compactness: poison data should be similar
            3. Coherence: The trigger’s perplexity should be low
            4. Target generation: triggers cause target action
    2. A practical memory injection attack against llm agents [[arxiv'25/05](https://arxiv.org/abs/2503.03704)]
    3. PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models [[USENIX Security'25/07](https://arxiv.org/abs/2402.07867)]
6. **Tool poisoning**
    1. MCP Security Notification: Tool Poisoning Attacks
        1. Jumping the line: How MCP servers can attack you before you ever use them [[blog'25/04](https://blog.trailofbits.com/2025/04/21/jumping-the-line-how-mcp-servers-can-attack-you-before-you-ever-use-them/)]
        2. How MCP servers can steal your conversation history [blog'25/04](https://blog.trailofbits.com/2025/04/23/how-mcp-servers-can-steal-your-conversation-history/)
    2. Prompt Injection Attack to Tool Selection in LLM Agents [[arxiv'25/08](https://arxiv.org/pdf/2504.19793)]
7. **Exfiltration attack: Inject URL to exploit renderers that fetches data from attacker’s server, leaking agent data**
    1. When Public Prompts Turn Into Local Shells: ‘CurXecute’ – RCE in Cursor via MCP Auto‑Start (EchoLeak cve-2025-32711) [[blog'25/08](https://www.aim.security/lp/aim-labs-echoleak-blogpost)]
    2. Simon Willison’s Weblog tagged exfiltration-attacks [[https://simonwillison.net/tags/exfiltration-attacks/](https://simonwillison.net/tags/exfiltration-attacks/)]
8. Some related works that attack the instruction following of LLMs (related to agents but mainly about model)
    1. An LLM Can Fool Itself: A Prompt-Based Adversarial Attack [[ICLR'24/05](https://arxiv.org/pdf/2310.13345)]
        1. Audit the LLM’s adversarial robustness via a prompt-based adversarial attack
        2. Let LLMs generate adversarial prompts, and define the generation prompts with three components:
            1. original input (OI), including the original sample and its ground-truth label
            2. attack objective (AO) illustrating a task description of generating a new sample that can fool itself without changing the semantic meaning
            3. attack guidance (AG) containing the perturbation instructions, e.g., add some characters 
    2. The SIFo Benchmark: Investigating the Sequential Instruction Following Ability of Large Language Models (Testing-phase backdoor) [[EMNLP'24/11](https://arxiv.org/pdf/2406.19999)]
        1. text modification, question answering, mathematics, and *security rule following*
    3. Can LLMs Follow Simple Rules? (Instruction following) [[arxiv'24/03](https://arxiv.org/pdf/2311.04235)]
        1. Propose Rule-following Language Evaluation Scenarios (RULES), a programmatic framework for measuring rule-following ability in LLM
        2. Defense: test-time steering and finetuning
    4. A Trembling House of Cards? Mapping Adversarial Attacks against Language Agents [[arxiv'24/12](https://arxiv.org/pdf/2402.10196)]
    5. Misusing Tools in Large Language Models With Visual Adversarial Examples [[ICLR'24/05](https://arxiv.org/abs/2310.03185)]
        1. Visual input-based prompt injection (applicable to both direct and indirect prompt injections)


### Attack against specific agents



1. ChatGPT Operator:
    1. ChatGPT Operator: Prompt Injection Exploits & Defenses [[blog'25/02](https://embracethered.com/blog/posts/2025/chatgpt-operator-prompt-injection-exploits/)]
    2. How ChatGPT Remembers You: A Deep Dive into Its Memory and Chat History Features [[blog'25/05](https://embracethered.com/blog/posts/2025/chatgpt-how-does-chat-history-memory-preferences-work)]
2. **Web agents**: Most attacks manipulate the web with malicious contents/queries, when LLM agents interact with the web, the malicious contents will be fed to the agents. Some attacks can be applied to multi-modal (MM) agents but the malicious contents are only in text
    1. Dissecting Adversarial Attacks on Multimodal LM Agents [[ICLR'25/04](https://arxiv.org/abs/2406.12814)]
        1. Classify attacks into:
            1. Illusioning: maintain the original user task while subtly manipulating information retrieved from tools, e.g., for a shopping agent, the user asks the agent to buy the cheapest jacket, then a malicious seller can inject a prompt that indicates its price is the lowest to mislead the lim.
            2. Goal misdirection: ask the agent to ignore the user task and follow the injected prompt
        2. Attack method: manipulation of uploaded item images/texts
    2. EIA: Environmental Injection Attack on Generalist Web Agents for Privacy Leakage [[ICLR'25/04](https://arxiv.org/abs/2409.11295)]
        1. Threat models
            1. Leak users' PII or users' task (let LLM fill PII into an invisible box)
            2. The web developer is malicious or malicious users contaminate development tools
        2. Inject persuade prompt into the HTML content of webpages
        3. If the attacker controls the web, it can get PII directly; no need to do it through injection
    3. AdvAgent: Controllable Blackbox Red-teaming on Web Agents [[ICML'25/07](https://arxiv.org/pdf/2410.17401)]
        1. Similar threat model as EIA but automatically generates attack prompts 
    4. Wipi: A new web threat for LLM-driven web agents [[arxiv'24/12](https://arxiv.org/abs/2402.16965)]
        1. Similar paper with EIA. but targets on llm with rag instead of web agent
        2. Aim to disrupt llm by using misdirect (e.g. Don’t summarize the webpage content)
    5. Prompt-to-SQL Injections in LLM-Integrated Web Applications: Risks and Defenses [[ICSE'25/04](https://www.computer.org/csdl/proceedings-article/icse/2025/056900a076/215aWuWbxeg)]
    6. CVE-Bench: A Benchmark for AI Agents’ Ability to Exploit Real-World Web Application Vulnerabilities [[ICML'25/07](http://arxiv.org/pdf/2503.17332)]
3. **Coding agent**
    1. A New Era in LLM Security: Exploring Security Concerns in Real-World LLM-based Systems (2024) [[arxiv'24/02](https://arxiv.org/abs/2402.18649)]
        1. Threat models: Prompt injection  
        2. LLM system (agent): objects; actions (information processing execution for individual objects); interactions; constraints
        3. Analysis of constraints of actions (more like single-actor conversations)
            1. Case study: LLM Outputs External Image Links with Markdown Format
            2. Existence of constraints: yes & robustness of constraints: not that robust (bypass via jailbreaking and indirect prompt injection)
        4. Analysis of constraints of interactions
            1. Cases: Sandbox (unauthorized access)/Web tools (indirect plugin calling)/Frontend (render malicious URLs)
        5. End2end attacks (exploit chain): the attack goal is to steal users’ private conversation records  
    2. RedCode: Risky Code Execution and Generation Benchmark for Code Agents [[NeurIPS'24/12](https://arxiv.org/abs/2411.07781)]
        1. Benchmark coding model/agent risks: generating malware and executing malicious stuff
    3. RedCodeAgent: Automatic Red-teaming Agent against Code Agents [[openreview paper'25/02](https://openreview.net/pdf?id=Mvn5g49RrM)]
        1. Threat model: Mislead the LLM agent into using a specified tool (third attack type)
        2. An upgrade of RedCode-Exec with the refinement capabilities 
        3. Use RedCode-Exec dataset for risk scenario and requirement
    4. Abuse code interpreters to gain access to underlying host operating systems or use them as a platform to wage cyber attacks 
        1. CYBERSECEVAL 2: A Wide-Ranging Cybersecurity Evaluation Suite for Large Language Models [[arxiv'24/04](https://arxiv.org/pdf/2404.13161)]
            1. Test whether a target code generation models/LLM refuse to execute malicious requests 
                1. Construct 500 interpreter abuse prompts 
                2. Use another LLM to judge whether the target model is compliant with the malicious request or refuses the malicious request
        2. Defense: Use high-quality data to train (identified by previous generation of models); safety evaluation and tuning; lower benign refusals (finetune with high-quality data)
    5. Some CVEs (arbitrary code execution and SQL injection)
        1. CVE Record of prompt injection [https://cve.mitre.org/cgi-bin/cvekey.cgi?keyword=Prompt+injection](https://cve.mitre.org/cgi-bin/cvekey.cgi?keyword=Prompt+injection) 
        2. When Prompts Go Rogue: Analyzing a Prompt Injection Code Execution in Vanna.AI [[blog'24/07](https://jfrog.com/blog/prompt-injection-attack-code-execution-in-vanna-ai-cve-2024-5565/)]
        3. RCE: Illegal Command Filter Bypass in langchain_experimental [https://github.com/langchain-ai/langchain/issues/21592](https://github.com/langchain-ai/langchain/issues/21592)
        4. LLM pentest: Leveraging agent integration for RCE [[blog'24/05](https://www.blazeinfosec.com/post/llm-pentest-agent-hacking/)]
        5. Impact of remote-code execution vulnerability in LangChain [[blog'23/07](https://www.ntietz.com/blog/langchain-rce/)]
    6. Demystifying RCE Vulnerabilities in LLM-Integrated Apps [[ACM CCS'24/10](https://arxiv.org/pdf/2309.02926)]
    7. ARTEMIS: Analyzing LLM Application Vulnerabilities in Practice [[Proceedings of the ACM on Programming Languages'25/04](https://dl.acm.org/doi/10.1145/3720488)]
4. **Personal assistant agents**
    1. Attacking Vision-Language Computer Agents via Pop-ups [[ACL'25/07](https://arxiv.org/abs/2411.02391)]
        1. Indirect prompt injection, attack through pop-ups
        2. Concern: pop-up is (maybe) easy to defend and easy to remove on the web
    2. Data Exposure from LLM Apps: An In-depth Investigation of OpenAI's GPTs [[IMC'25/10](https://arxiv.org/abs/2408.13247)]
    3. LLM Platform Security: Applying a Systematic Evaluation Framework to OpenAI's ChatGPT Plugins [AIES'24/10](https://arxiv.org/abs/2309.10254)
        1. Categorize and evaluate the risks of using untrusted third-party plugins 
    4. Personal LLM Agents: Insights and Survey about the Capability, Efficiency, and Security [[arxiv'24/05](https://arxiv.org/abs/2401.05459)]


## Blue-teaming

### Model-based defenses 



1. Guardrail
    1. Flexible, but can be potentially bypassed with guardrail injection and incurs computational costs [[arxiv'25/07](https://arxiv.org/abs/2504.11168)]
    2. ShieldAgent: Shielding Agents via Verifiable Safety Policy Reasoning [[ICML'25/07](https://shieldagent-aiguard.github.io/)]
        1. Guardrail policy generation based on documents and LLM
    3. A Holistic Approach to Undesired Content Detection in the Real World (OpenAI guardrail) [[AAAI'23/02](https://arxiv.org/abs/2208.03274)]
    4. Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations [[arxiv'23/12](https://arxiv.org/abs/2312.06674v1)]
        1. classification model on LLM inputs and outputs, have 6 safety-related categories
    5. Llama Prompt Guard 2 [[blog](https://www.llama.com/docs/model-cards-and-prompt-formats/prompt-guard/)]
        1. [meta-llama/Llama-Prompt-Guard-2-86M](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M)
        2. Llama Prompt Guard 2 models classify prompts as ‘malicious’ if the prompt explicitly attempts to override prior instructions embedded into or seen by an LLM. This classification considers only the intent to supersede developer or user instructions, regardless of whether the prompt is potentially harmful or the attack is likely to succeed.
    6. NeMo Guardrails: A Toolkit for Controllable and Safe LLM Applications with Programmable Rails [[EMNLP'23/12](https://arxiv.org/abs/2310.10501)]
        1. Define a DSL with a set of rules, calibrate input queries with rules based on embedding distance (with KNN)
        2. Input rail (user input), Dialog rail (next step), Retrieval rail (external data), Execution rail (tool call), Output rail (final output)
    7. Guardrails AI [[website](https://www.guardrailsai.com/)]
        1. User-specified rail specs; guard (regular expression or classifier); if trigger error, correct the user prompts; output checking
    8. Position: building guardrails for large language models requires systematic design [[ICML'24/07](https://dl.acm.org/doi/abs/10.5555/3692070.3692521)]
        1. Suggestions: formal guarantee to avoid arms races; resolving conflicts via priority and ensemble; rule-based & learning-based solutions
    9. A Causal Explainable Guardrails for Large Language Models (LLMGuardrail) [[ACM CCS'24/12](https://dl.acm.org/doi/10.1145/3658644.3690217)]
    10. WebGuard: Building a Generalizable Guardrail for Web Agents [[arxiv'25/07](https://arxiv.org/abs/2507.14293)]
2. Inference-phase defenses: 
    1. Finetune a classifier to identify prompt injection (similar as guardrail)
        1. Fine-Tuned DeBERTa-v3-base for Prompt Injection Detection: 
            1. [protectai/deberta-v3-base-prompt-injection-v2](http://huggingface.co/ProtectAI/deberta-v3-base-prompt-injection-v2)
        2. GenTel-Safe: A Unified Benchmark and Shielding Framework for Defending Against Prompt Injection Attacks [[arxiv'24/09](https://arxiv.org/abs/2409.19521)]
            1. Finetune a classifier (which is independent from the LLM) to detect the model
        3. Embedding-based classifiers can detect prompt injection attacks [[CAMLIS'24/10](https://arxiv.org/pdf/2410.22284)]
            1. Use a pretrained embedding model to embed benign prompts and prompt injection attacked prompts, then use traditional machine learning method for classification (regression, xgboost, etc)
        4. DataSentinel: A Game-Theoretic Detection of Prompt Injection Attacks [[IEEE S&P'25/05](https://arxiv.org/abs/2504.11358)]
            1. Identify prompt injection in untrusted inputs by leveraging a fine-tuned model vulnerable to prompt injection, with known-answer detection
    2. Prompt-level defense: 
        1. Defense Against Prompt Injection Attack by Leveraging Attack Techniques [[ACL'25/07](https://arxiv.org/abs/2411.00459)]
            1. Use prompt injection attacks to append the original benign prompt at the end of input prompts
        2. Formalizing and Benchmarking Prompt Injection Attacks and Defenses [[USENIX Security'24/08](https://www.usenix.org/conference/usenixsecurity24/presentation/liu-yupei)]
            1. known-answer detection (append an additional instruction into user prompt, e.g., “say a secret word xxx”), then detect if outputs contain this secret word
        3. PromptShield: Deployable Detection for Prompt Injection Attacks [[ACM CODASPY'25/06](https://arxiv.org/abs/2501.15145)]
        4. Robustness via Referencing: Defending against Prompt Injection Attacks by Referencing the Executed Instruction [[arxiv'25/04](https://arxiv.org/abs/2504.20472)]
        5. FATH: Authentication-based Test-time Defense against Indirect Prompt Injection Attack [[arxiv'24/11](https://arxiv.org/abs/2410.21492)]
            1. Runtime-generated authentication code
        6. Defending Against Indirect Prompt Injection Attacks With Spotlighting (Microsoft Spotlighting) [[arxiv'24/03](https://arxiv.org/abs/2403.14720)]
            1. Add delimeters
    3. Defense based on internal representations 
        1. Get my drift? Catching LLM Task Drift with Activation Delta [[IEEE SaTML'25/04](https://arxiv.org/abs/2406.00799)]
        2. Attention Tracker: Detecting Prompt Injection Attacks in LLMs [[NAACL'25/04](https://aclanthology.org/2025.findings-naacl.123.pdf)]
            1. intuition: if a malicious prompt is injected, the attention score over the original prompt will largely decrease
            2. Find the “important attention head” that behaves differently under benign prompts and prompt injection attack prompts, summing the attention score over the original prompt tokens, if less than a certain threshold, deny to respond
3. Training-based defenses:
    1. StruQ: Defending Against Prompt Injection with Structured Queries [[USENIX Security'25/08](https://www.usenix.org/system/files/conference/usenixsecurity25/sec24winter-prepub-468-chen-sizhe.pdf)]
        1. Build structured prompts that have special tokens to separate user prompts and user data
        2. Finetune the model to ignore contents after the specific tokens
    2. SecAlign: Aligning LLMs to Be Robust Against Prompt Injection [[arxiv'25/07](https://arxiv.org/abs/2410.05451v1)]
        1. Preference learning-based finetuning: The LLM is only trained to favor the desirable response, but does not know what an undesirable response looks like. Thus, a secure LLM should also observe the response to the injected instruction and be steered away from that response.
    3. Rule-Based Rewards for Language Model Safety [[NeurIPS'24/12](https://arxiv.org/abs/2411.01111)]
        1. Better alignment strategy
    4. Jatmo: Prompt Injection Defense by Task-Specific Finetuning [[ESORICS'24/09](https://dl.acm.org/doi/abs/10.1007/978-3-031-70879-4_6)]
    5. The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions [[arxiv'24/04](https://arxiv.org/abs/2404.13208)]
        1. Construct data for both “aligned” instructions and “misaligned” instructions
        2. fine-tuning protocol that induces an LLM to privilege higher privilege instructions over lower-privilege instructions (system > developer > user).
    6. Instructional Segment Embedding: Improving LLM Safety with Instruction Hierarchy [[ICLR'25/04](https://arxiv.org/abs/2410.09102)]
        1. Adding a segment embedding layer for better learning instruction hierarchy



### System-level Runtime Defense



1. **Input validation and sanitization: Guardrails on LLM input**
    1. Benefits: Guardrails’ non-invasiveness allows minor modification and utility impact in the agent
    2. Limitations: Challenging to provide resilience against adaptive attacks 
    3. Model-based guardrails
        1. LLlamaFirewall: An open source guardrail system for building secure AI agents [[arxiv'25/05](https://arxiv.org/abs/2505.03574)]
            1. PromptGuard: detect direct/indirect prompt injection
        2. Microsoft Prompt Shields [[website](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/jailbreak-detection)]
            1. detect safety and security of LLM output (e.g., hate, violence, self-harm, direct/indirect prompt injection)
    4. Rule-based guardrails
        1. Nvidia NeMo: DSL for input guardrail [[github repo](https://github.com/NVIDIA-NeMo/NeMo)]
        2. Google safe browsing [[website](https://safebrowsing.google.com/)]
        3. Content Security Policy (CSP) allow-list [[website](https://content-security-policy.com/)]
        4. Building a secure agentic ai application leveraging a2a protocol (Google) [[arxiv'25/04](https://arxiv.org/abs/2504.16902)]
2. **Policy enforcement: Guardrails on LLM output**
    1. Benefits: Guardrails’ non-invasiveness allows minor modification and utility impact in the agent
    2. Limitations: Challenging to provide resilience against adaptive attacks 
    3. Model-based guardrails
        1. LLlamaFirewall: An open source guardrail system for building secure AI agents [[arxiv'25/05](https://arxiv.org/abs/2505.03574)]
            1. AlignmentCheck: LLM output alignment check.
        2. Microsoft Prompt Shields [[website](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/jailbreak-detection)]
            1. detect safety and security of LLM output (e.g., hate, violence, self-harm, direct/indirect prompt injection)
    4. Rule-based guardrails
        1. Nvidia NeMo: DSL for LLM output guardrail [[github repo](https://github.com/NVIDIA-NeMo/NeMo)]
        2. Mitigating prompt injection attacks with a layered defense strategy (Google) [[blog'25/07](https://security.googleblog.com/2025/06/mitigating-prompt-injection-attacks.html)]
            1. UI renderer specific : Only render google internal images
        3. LLlamaFirewall: An open source guardrail system for building secure AI agents [[arxiv'25/05](https://arxiv.org/abs/2505.03574)]
            1. CodeShield: regex based rules to detect malicious LLM-generated code
        4. AI Agents with Formal Security Guarantees [[ICML Workshop of Next Generation of AI Safety'24/07](https://openreview.net/pdf?id=c6jNHPksiZ)]
            1. Provide policy language for agents that can be manually defined by agent developers.
            2. Information flow policy
    5. Hybrid guardrails: Generate rules with model or agent
        1. GuardAgent: Safeguard LLM Agents via Knowledge-Enabled Reasoning [[ICML'25/07](https://arxiv.org/abs/2406.09187)]
            1. Agent for Guardrail generation. Generate guardrails using LLM and code execution/debugging tools.
            2. Similar to Progent.
        2. AGrail: A Lifelong Agent Guardrail with Effective and Adaptive Safety Detection [[ACL'25/07](https://aclanthology.org/2025.acl-long.399.pdf)]
            1. Use model to generate safety checks
            2. Use model and tools to perform safety checks before action
        3. Progent: Programmable Privilege Control for LLM Agents [[arxiv'25/04](https://arxiv.org/pdf/2504.11703)]
            1. Design a runtime fine-grained policy generation framework and automate the policy generation with LLM
            2. Runtime agent sandboxing: Constrain agent tool calls based on the previous context.
        4. Contextual Agent Security: A Policy for Every Purpose [[ACM HotOS'25/05](https://dl.acm.org/doi/10.1145/3713082.3730378)]
            1. Same as Progent. Runtime policy generation based on trusted context data (i.e., Contextual policy) and fine-grained agent sandbox enforcement.
    10. AgentSpec: Customizable Runtime Enforcement for Safe and Reliable LLM Agents [[ICSE'26/04](https://arxiv.org/abs/2503.18666)]
3. **Identity and privilege management**
    1. Agent identity
        1. Motivation: Visibility to regulators, users (whether they’re interacting with agents or human), runtime monitoring, post-hoc analysis
        2. Visibility into AI agents (measures to improve agent visibility) [[ACM FAccT'25/06](https://arxiv.org/pdf/2401.13138)]
            1. Agent identifiers: Agent card, Underlying system, Involved actors -- clarify who is accountable in case an agent causes harm.
            2. Real-time monitoring
            3. Activity logs
        3. IDs for AI systems [[NeurIPS Workshop of Regular ML'24/12](https://arxiv.org/abs/2406.12137)]
        4. Infrastructure for AI Agents [[TMLR'25/05](https://arxiv.org/abs/2501.10114)]
            1. Attribution: Identity binding, Certification, Agent IDs
            2. Interaction: Agent network channels, Oversight layers (e.g., user intervention), inter-agent communication, commitment devices
            3. Response: Incident reporting, rollbacks
        5. Prompt Infection: LLM-to-LLM Prompt Injection within Multi-Agent Systems [[arxiv'24/10](https://arxiv.org/abs/2410.07283v1)]
            1. LLM tagging - `[Agent name]: ` in front of agent responses.
    2. Centralized identity
        1. okta - general identity management [[website](https://www.okta.com/identity-101/identity-and-access-management/)]
        2. composio - agent identity management [[website](https://composio.dev/agentauth)]
        3. OpenID-Connect protocol (OIDC) [[website](https://www.microsoft.com/en-us/security/business/security-101/what-is-openid-connect-oidc)]
    3. Decentralized identity
        1. Agent Network Protocol - Identity and Encrypted Communication Layer [[github repo](https://github.com/agent-network-protocol/AgentNetworkProtocol/tree/main)]
            1. W3C DID (Decentralized Identifiers) [[website](https://www.w3.org/TR/did-1.1/)]
        2. Microsoft Verified ID [[website](https://learn.microsoft.com/en-us/entra/verified-id/decentralized-identifier-overview)]
    4. Agent privilege management
        1. Authenticated Delegation and Authorized AI Agents [[ICML'25/07](https://arxiv.org/abs/2501.09674)]
            1. human user creating a digital authorization that a specific AI agent can use to access a digital service (or interact with another AI agent) on behalf of the user, which can be verified by the corresponding service or agent for its authenticity.
            2. A delegation token authorizes an AI agent to act on the user’s behalf
            3. Task scoping and resource scoping
        2. Composio [[github repo](https://github.com/ComposioHQ/composio)]
            1. User-friendly agent framework that provides OAuth token-based IAM
        3. robots.txt [[website](https://www.robotstxt.org/)]
    5. RAG/VectorDB Access control
        1. Access control for vector stores using metadata filtering with Amazon Bedrock Knowledge Bases [[blog'24/07](https://aws.amazon.com/blogs/machine-learning/access-control-for-vector-stores-using-metadata-filtering-with-knowledge-bases-for-amazon-bedrock/)]
        2. HoneyBee: Efficient Role-based Access Control for Vector Databases via Dynamic Partitioning [[arxiv'25/05](https://arxiv.org/abs/2505.01538)]
            1. Find a balance between space-time tradeoff for vectorDB access control using dynamic partitioning
        3. ControlNet: A Firewall for RAG-based LLM System [[arxiv'25/04](https://arxiv.org/abs/2504.09593)]
            1. RAG access control with LLM activation-based detection and mitigation
4. **Privilege separation and access control**
    1. LLM Agents Should Employ Security Principles [[arxiv'25/05](https://arxiv.org/abs/2505.24019)]
        1. Defense-in-depth strategy: Least privilege by dividing agents into persistent/ephemeral agents. Complete mediation by data minimizer and response filter.
        2. Automated policy configuration by a reward modeling policy engine: Adaptive policy learning based on task success rate.
    2. Planner-Executor separation
        1. Defeating Prompt Injections by Design (CaMeL) [[arxiv'25/06](https://arxiv.org/abs/2503.18813)]
            1. Separate privileges of an agent: planner agent generates a program (i.e., determines control flow and data flow), and a quarantine agent parses untrusted data. Prevents untrusted data from corrupting the control and data flow.
            2. Data flow policies
        2. Securing AI Agents with Information-Flow Control (FIDS) [[arxiv'25/05](https://arxiv.org/abs/2505.23643)]
        3. System-Level Defense against Indirect Prompt Injection Attacks: An Information Flow Control Perspective (f-secure) [[arxiv'24/10](https://arxiv.org/abs/2409.19091)]
            1. Control the information flows and access control in an agent system to prevent malicious information from being propagated and executed by the agent system
            2. Disaggregates the components of an LLM system into a context-aware pipeline with dynamically generated structured executable plans, and a security monitor filters out untrusted input into the planning process 
            3. Provide formal models with an analysis of the security guarantee 
            4. Privilege separation: planner and unprivileged parser.
        4. Design Patterns for Securing LLM Agents against Prompt Injections [[arxiv'25/06](https://arxiv.org/abs/2506.08837)]
    3. Agent privilege separation
        1. Prompt Flow Integrity to Prevent Privilege Escalation in LLM Agents [[arxiv'25/04](https://arxiv.org/abs/2503.15547)]
            1. Separate privileges of an agent: a privileged agent and an unprivileged agent have different tool permissions, managed by their access tokens.
            2. Prevent confused deputy: i) Replace untrusted data from the lower-privilege agent into a data ID, so that they cannot corrupt control/data flow of the privileged agent. ii) Track untrusted data to prevent it from being used in unsafe data flow.
    4. Security module - Executor separation
        1. AirGapAgent: Protecting Privacy-Conscious Conversational Agents [[ACM CCS'24/10](https://dl.acm.org/doi/10.1145/3658644.3690350)]
            1. Design a runtime data minimization to prevent prompt injection attacks that leak confidential data
            2. Separate privileges of an agent: a data minimization agent that selects privacy data based on trusted data, and the baseline agent that handles untrusted data with the minimized privacy data.
    5. Mobile system privilege separation
        1. SecGPT: An Execution Isolation Architecture for LLM-Based Systems (IsolateGPT) [[NDSS'25/02](https://arxiv.org/abs/2403.04960)]
            1. Design interfaces and permission control to Isolate the execution of GPT-integrated third-party apps
            2. Target attacks: app compromise, data stealing, inadvertent data exposure, and uncontrolled system alteration
5. **Monitoring and auditing**
    1. GUARDIAN: Safeguarding LLM Multi-Agent Collaborations with Temporal Graph Modeling [[arxiv'25/05](https://arxiv.org/abs/2505.19234)]
        1. Detect error propagation in multi-agent system with graph modeling
    2. SentinelAgent: Graph-based Anomaly Detection in LLM-based Multi-Agent Systems [[arxiv'25/05](https://arxiv.org/abs/2505.24201)]
        1. Represents the multi-agent system (MAS) as an interaction graph with nodes (agents/tools) and edges (communications).
        2. Global anomaly detection: task-level output divergence, prompt-level attacks
        3. Single-point failure localization: specific faulty agents/tools, tool misuse
        4. Multi-point failure attribution: distributed or emergent issues, 
    3. Testing Language Model Agents Safely in the Wild [[NeurIPS Workshop of Socially Responsible Language Modelling Research'23/12](https://arxiv.org/abs/2311.10538)]
        1. Monitoring harmful or offtask agent output in web and file access 
        2. web search, browse website, write to file, read file, list files, execute python file, and execute python code.
    4. Disclosure Audits for LLM Agents [[arxiv'25/06](https://www.arxiv.org/pdf/2506.10171)]
        1. Auditing for accumulatively steering conversation to induce privacy leakage.
        2. Detects explicit leakage with LLM judge and implicit leakage
    5. AgentAuditor: Human-Level Safety and Security Evaluation for LLM Agents [[arxiv'25/05](https://arxiv.org/abs/2506.00641)]
        1. Extracts structured features (scenario, risk, behavior) from agent interactions and constructs RAG. Reference relevant examples to assess new agent interactions.
    6. Monitoring LLM Agents for Sequentially Contextual Harm [[ICLR Workshop of Building Trust'25/04](https://openreview.net/pdf?id=LC0XQ6ufbr)]
        1. Task decomposition: Seemingly benign subtasks for high-level malicious task
        2. adaptive attacks with task decomposition can bypass existing guardrails or LLM-based monitors.
    7. Visibility into AI Agents [[ACM FAccT'25/06](https://arxiv.org/pdf/2401.13138)]
        1. Real-time monitoring without logs and activity logs for post-hoc analysis, forensic investigation
6. **Information flow control, taint tracking**
    1. Challenges
        1. How to track data flow in an LLM agent?
        2. What data flow policies to enforce?
    2. Multi-execution-based data flow tracking
        1. Permissive Information-Flow Analysis for Large Language Models [[arxiv'25/05](https://arxiv.org/abs/2410.03055)]
            1. Permissive IFC in LLM: Secure multi-execution [[IEEE S&P'10/05](https://ieeexplore.ieee.org/document/5504711)] for information flow analysis in LLM
        2. MELON: Provable Defense Against Indirect Prompt Injection Attacks in AI Agents [[ICML'25/07](https://arxiv.org/abs/2502.05174)]
            1. Detect indirect prompt injection by measuring the original user prompt and the task-neutral prompt. Detects attacker-injected tool calls.
    3. Symbolization-based data flow tracking
        1. Prompt Flow Integrity to Prevent Privilege Escalation in LLM Agent [[arxiv'25/04](https://arxiv.org/abs/2503.15547)]
            1. Taint tracking to prevent privilege escalation and confused deputy attack
        2. Defeating Prompt Injections by Design (CaMeL) [[arxiv'25/06](https://arxiv.org/abs/2503.18813)]
            1. Taint tracking to ensure external system policy compliance
        3. Securing AI Agents with Information-Flow Control (FIDS) [[arxiv'25/05](https://arxiv.org/abs/2505.23643)]
            1. IFC to enforce confidentiality and integrity at the same time
        4. RTBAS: Defending LLM Agents Against Prompt Injection and Privacy Leakage [[arxiv'25/02](https://arxiv.org/abs/2502.08966)]
            1. IFC combined with a model-based dependency screener to overcome label creep
    4. LLM-based control/data dependency analysis
        1. AgentArmor: Enforcing Program Analysis on Agent Runtime Trace to Defend Against Prompt Injection [[arxiv"25/08](https://www.arxiv.org/abs/2508.01249)]
    5. Policies
        1. Prompt Flow Integrity to Prevent Privilege Escalation in LLM Agents [[arxiv'25/04](https://arxiv.org/abs/2503.15547)]
            1. Privilege escalation: PFI 
        2. Securing AI Agents with Information-Flow Control (FIDS) [[arxiv'25/05](https://arxiv.org/abs/2505.23643)]
            1. Confidentiality and Integrity
        3. Defeating Prompt Injections by Design (CaMeL) [[arxiv'25/06](https://arxiv.org/abs/2503.18813)]
            1. Policy compliance
7. **Formal verification**
    1. Formal modelling agent (Not for security purpose)
        1. Planning Anything with Rigor: General-Purpose Zero-Shot Planning with LLM-based Formalized Programming [[ICLR'25/04](https://arxiv.org/pdf/2410.12112)]
            1.  formally formulate and solve them as optimization problems to improve planning performance of LLM
        2. PDL: A Declarative Prompt Programming Language [[arxiv'24/10](https://arxiv.org/abs/2410.19135)]
            1. make prompt programming simpler, less brittle, and more enjoyable
        3. Formally Specifying the High-Level Behavior of LLM-Based Agents [[arxiv'24/01](https://arxiv.org/abs/2310.08535)]
            1. declarative agent framework, such that the user specifies the desired high-level behavior in terms of constraints without concern for how they should be implemented or enforced
            1. Improves controllability and performance
        4. Formal-LLM: Integrating Formal Language and Natural Language for Controllable LLM-based Agents [[arxiv'24/08](https://arxiv.org/abs/2402.00798)]
            1. Framework that allows agent developers to express their requirements or constraints for the planning process as an automaton
            2. Improves controllability and performance
    2. Mobile GUI agent verification
        1. Safeguarding Mobile GUI Agent via Logic-based Action Verification [[arxiv'25/03](https://arxiv.org/abs/2503.18492)]
            1. Define DSL to represent the desired behavior (user intent) and the actual behavior (app execution) in a unified, logically verifiable manner.
8. **Credential and secret management**
    1. Big Help or Big Brother? Auditing Tracking, Profiling, and Personalization in Generative AI Assistants [[USENIX Security'25/08](https://www.usenix.org/conference/usenixsecurity25/presentation/vekaria?utm_source=chatgpt.com)]
    2. Incidents
        1. ChatGPT banned in Italy over privacy concerns [[blog'23/04](https://www.bbc.co.uk/news/technology-65139406)]
        2. March 20 ChatGPT outage: Here’s what happened (ChatGPT chat history leakage in March 2023) [[blog'23/03](https://openai.com/index/march-20-chatgpt-outage/)]
            1. Due to a vulnerability in Redis server - Request use-after-free
        3. Meta: Help Users Stop Accidentally Sharing Private AI Chats (Meta AI fails to show privacy notice to users) [blog'25/07](https://www.mozillafoundation.org/en/campaigns/meta-help-users-stop-accidentally-sharing-private-ai-conversations/)
    3. Services
        1. ChatGPT temporary chat [[website](https://help.openai.com/en/articles/8914046-temporary-chat-faq)]
            1. Temporary Chats won’t appear in your history, and ChatGPT won’t remember anything you talk about. For safety purposes we may still keep a copy for up to 30 days.


### Others

**Tool protection**

1. MCP Safety Audit: LLMs with the Model Context Protocol Allow Major Security Exploits [[arxiv'25/04](https://arxiv.org/abs/2504.03767)]

**Post-detection defenses**

1. User alert
2. Recovery
    2. GoEX: Perspectives and Designs Towards a Runtime for Autonomous LLM Applications [[arxiv'24/04](https://arxiv.org/abs/2404.06921)]
        1. Design post-facto validation with an undo feature and damage confinement  
        2. Access control (secret data is stored locally, ask for user’s permission), symbolic credentials, and sandboxing 
3. Logging and analysis

**Other references**

1. OWASP LLM Prompt Injection Prevention Cheat Sheet [[website](https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html)]
2. An Introduction to Google’s Approach to AI Agent Security [[website](https://research.google/pubs/an-introduction-to-googles-approach-for-secure-ai-agents/)]
    1. A hybrid defense-in-depth approach: combines traditional, deterministic security with dynamic, reasoning-based defenses
    2. Runtime policy enforcement (traditional, deterministic) limits the worst-case impact of agent malfunction.
    3. Reasoning-based solutions, including adversarial training, guard models, and security analysis.
3. Mitigating prompt injection attacks with a layered defense strategy (Google GenAI Security Team) [[blog'25/06](https://security.googleblog.com/2025/06/mitigating-prompt-injection-attacks.html)]
    1. Defense-in-depth approach
        1. Prompt injection content classifiers
        2. Security thought reinforcement
        3. Markdown sanitization and suspicious URL redaction
        4. User confirmation framework
        5. End-user security mitigation notifications

# Contributors

We thank the following awesome contributions: Zhun Wang, Kaijie Zhu, Yuzhou Nie, Tianneng Shi, Juhee Kim, Zeyi Liao, Ruizhe Jiang and Wenbo Guo (😄). Thank you!
