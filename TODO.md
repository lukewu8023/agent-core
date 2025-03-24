# TODO

## Feature v1
## trace
- graph first step to node (done)
- plan add adj (done)
- replan divided success and false (done)
- DEFAULT_SUCCESS_REPLAN_PROMPT 184 plan enhancement (done)
- SUCCESS PLAN current_attempts not 0 (done, no need fix, current_attempts will change base on process)
- Final Step action change next to end (done)
- implement a util to convert execution_trace to mermaid flowchat
- Generic planner execute history

### bug
- return json has "/" will cause crash

### langgragh
- visulizaiton follow langgragh diagram

## Feature v2
- change context by step(v2)
- knowledge & background rag(v2)
- knowledge graph to collect sufficient information(v2)
- human in the loop(v2)
- minimize context based on data(v2)

### Review
- review **Plan & Execution (Each node with execution results, if executed)** in success replan
- review **Failed History** in agent execution
- review **Failure Reason**
- review **Execution History**

## Test
- more unit testing
- test replan (breakdown done)
- test retry with same result

## Log & Doc
- better describe the 3 ways to define plan
- add trace example for replan and validation
- read me for each example
- add FAQ section

# Completed
- generic model creation and config
- stream processing for each step during the execution
- step validator by category, support customized
- history planning experience for improving
- successful generation rate
- put example in another project
- UI interaction
- pass parent task for all step execution 
- fix background used in planner
- validation add execution history 
- expose generic planner prompt from graph planner 
- change validator to evaluator
- abstract validator base, unique output return
- adjust planner prompt to use tool better
- add R2D2 in readme
- TODO list
- better describe the knowledge and background

- plan: background, knowledge, categories_str, task, tool
- execute: background, context, tool
- evaluator: background, context
- replan: background, knowledge, categories_str, task, tool
- summary include agent execute result
- compatible with langgraph
- compatible with autogen
- R2D2 guidance
- tool execute try catch exception add evaluator, add suggestion (v1) 
- validation retry in generic planner(v1) 

- Raw LLM request - log
- 4 times retry after success replan
- review replanning history
- only A.1 in context? why A.1.2.1 add even it failed
- 0.9 still rerun
- data issue in one step due to check component after not able to pass generic validation (done with only check tool calling not content)
- evaluator for tool, only check valid arguments and signature (done need test)
- evaluator output structure (done need test)
- Auto generate evaluation threashold (done need test)
- unique threshold pass into calculation (done need test)
- Auto generate missing step (especially new tool call) (done need test)
- reasoning and validation prompt in execute history(done need test)
- final response of the task(done need test)
- print evaluator suggestion(done need test)
- enrich llm related log (llm input/output/prompt)(done need test)
- centrilize output structure(done need test)
- enrich llm related log (llm input/output/prompt)
- execution loop control
- support overwrite validation max attempt and threshold
- debug for steps 
- input/output token (done)
- tool args in steps (done)
- output summary used llm(add in steps) (done)
- where this is from  INFO:Agent:Generating execution result summary. (done, get_execution_result_summary, because invoked summary method)
- Need to add execution result in **Plan & Execution (Each node with execution results, if executed)** in success replan (done)
- Need all executed results in **Plan & Execution (Each node with execution results, if executed)** in success replan (done)
- Need to add all failed execution in **Failed History** in agent execution (done)
- Wrong failure reason - **Failure Reason** 0.85 (done, change to suggestion)
- review **Execution History** (Notes: 1.0 is the full score. The closer to 1.0, the closer to accuracy. Less than evaluation_threshold mark as failed.) (done, deleted)
- refactor execution_trace structure (done)
- generate reasoing history real time based on execution_trace structure (*) (done)