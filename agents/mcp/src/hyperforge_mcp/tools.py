from hyperforge import PROMPT_ENVIRONMENT

PROMPT_CHOOSE = """
Choose proper prompt to solve the task. If none of the prompts is suitable, do not choose any. Make sure to choose only one prompt.

{% for prompt in prompts %}
<prompt name="{{prompt.name}}">{{prompt.description}}</prompt>
{% endfor -%}

Task: {{task_description}}

"""

PROMPT_CHOOSE_TEMPLATE = PROMPT_ENVIRONMENT.from_string(PROMPT_CHOOSE)

TOOLS_CHOOSE = """
Choose proper toolset to solve the task. If none of the toolsets is suitable, do not choose any. Make sure to choose only one toolset.

{% for tool in tools %}
<tool name="{{tool.name}}">{{tool.description}}</tool>
{% endfor -%}

Task: {{task_description}}

"""

TOOLS_CHOOSE_TEMPLATE = PROMPT_ENVIRONMENT.from_string(TOOLS_CHOOSE)

SYSTEM_SUMMARIZE_TOOLS = """
Context: There are many tools available for a user. However, the number of tools can be large, and it is not always practical to present all of them at once. We need to create a summary of them that accurately reflects the capabilities they provide.

The user present you with the tools available to them, and you must create a summary of the tools that is accurate and comprehensive. The summary should include the capabilities of the tools and when they should be used."""


TOOLS_SUMMARIZE_EXAMPLES = """
{% for tool in tools %}
<tool name="{{mcp_id}}-{{tool.id}}">{{tool.description}}</tool>
{% endfor -%}


Your response must follow the JSON schema:

```
{
  "type": "object",
  "required": [
    "name",
    "summary"
  ],
  "properties": {
    "summary": {
      "type": "string",
      "description": "A summary of the tool capabilities, including their capabilities and how they can be used together. This may be up to five pararaphs long, be careful not to leave out important details.",
      "example": "These tools assist with authoring the \"foo\" language. They can provide diagnostics, run tests, and provide refactoring actions for the foo language."
    },
    "name": {
      "type": "string",
      "description": "A short name for the group. It may only contain the characters a-z, A-Z, 0-9, and underscores.",
      "example": "foo_language_tools"
    }
  }
}
"""

TOOLS_SUMMARIZE_EXAMPLES_TEMPLATE = PROMPT_ENVIRONMENT.from_string(
    TOOLS_SUMMARIZE_EXAMPLES
)
SIMPLE_TOOL_CHOICE_PROMPT = """"You are an agent that must choose the best tools to perform a task. If more than one tool is needed, return them all."
"""

SYSTEM_PROMPT = """You are an agent - please keep going until the user’s query is completely resolved, before ending your turn and yielding back to the user. Only terminate your turn when you are sure that the problem is solved, or if you need more info from the user to solve the problem.

If you are not sure about anything pertaining to the user’s request, use your tools to read files and gather the relevant information: do NOT guess or make up an answer.

You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls. DO NOT do this entire process by making function calls only, as this can impair your ability to solve the problem and think insightfully."""


MCP_ROUTER_PROMPT = """
<reminderInstructions>
You are an agent - you must keep going until the user's query is completely resolved, before ending your turn and yielding back to the user. ONLY terminate your turn when you are sure that the problem is solved, or you absolutely cannot continue.
You take action when possible- the user is expecting YOU to take action and go to work for them. Don't ask unnecessary questions about the details if you can simply DO something useful instead.
</reminderInstructions>

{% if userInformation %}
<userInformation>
{{user}}

</userInformation>

{% endif %}

{% if currentContext %}
<currentContext>
{{context}}

</currentContext>

{% endif %}
<userRequest>
{{question}}

</userRequest>

"""
MCP_ROUTER_PROMPT_TEMPLATE = PROMPT_ENVIRONMENT.from_string(MCP_ROUTER_PROMPT)
