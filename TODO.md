# TODO

### Feature v1
- compatible with autogen
- execution length limitation
- data issue in one step due to check component after not able to pass generic validation
- tool execute try catch exception add evaluator, add suggestion (v1) (done need test)
- validation retry in generic planner(v1) (done need test)
- how to debug in the library?
### Feature v2
- change context by step(v2)
- enrich log (input/output/prompt)(v2)
- knowledge & background rag (v2)
- support overwrite validation max attempt and threshold(v2)
- knowledge graph to collect sufficient information(v2)

### Test
- more unit testing
- test replan (breakdown done)

### Doc
- read me for each example
- R2D2 guidance
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
- bug (done)
```json
{
     "id": "B.2.1.1.2",
     "task_description": "Select the right wing emoji characters for the dragon, focusing on specific styles and sizes.",
     "next_nodes": ["B.3"],
     "validation_threshold": 0.9,
     "max_attempts": 3
}
```
- plan: background, knowledge, categories_str, task, tool
- execute: background, context, tool
- evaluator: background, context
- replan: background, knowledge, categories_str, task, tool
- summary include agent execute result
- compatible with langgraph