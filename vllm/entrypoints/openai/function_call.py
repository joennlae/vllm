import json
from typing import Union
from vllm.logger import init_logger
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolParam,
    ToolCallsDelta,
    ToolCallsMessage,
    FunctionCall,
    ChatCompletionAssistantMessage,
    ChatCompletionToolMessage,
)


logger = init_logger(__name__)


class OpenAIToolsPrompter:
    """
    https://platform.openai.com/docs/assistants/tools
    """

    def __init__(self):
        pass

    @classmethod
    def func_call_token_pre(cls) -> str:
        return "!"

    @classmethod
    def func_call_token_size(cls) -> int:
        return 15

    @classmethod
    def func_call_token(cls) -> str:
        return "!function_call:"

    def content_from_assistant(self, message: ChatCompletionAssistantMessage) -> str:
        text = ""
        for call in message.tool_calls:
            text += (
                call.id
                + " was called with arguments : "
                + str(call.function.arguments)
                + "\n"
            )
        if message.content is None:
            return text
        else:
            return message.content + "\n" + text

    def content_from_tool(self, message: ChatCompletionToolMessage) -> str:
        return message.tool_call_id + " -> " + message.content

    def inject_prompt(self, request: ChatCompletionRequest):
        """Tested with :
        https://huggingface.co/mlabonne/NeuralHermes-2.5-Mistral-7B/discussions/3"""
        if (
            request.tool_choice is not None
            and request.tools is not None
            and request.tool_choice == "auto"
        ):
            tools_list: [ChatCompletionToolParam] = request.tools
            if len(tools_list):
                text_inject = (
                    "Your task is to call a function when needed. "
                    "You will be provided with a list of functions. "
                    "\n\nAvailable functions:\n"
                )
                for tool in tools_list:
                    if tool.type == "function":
                        text_inject += "\n" + tool.function.name
                        if tool.function.description is not None:
                            text_inject += " - " + tool.function.description
                        if tool.function.parameters is not None:
                            schema = json.dumps(tool.function.parameters, indent=4)
                            text_inject += f"```\njsonschema\n{schema}\n```"
                text_inject += (
                    f"\nTo call a function, the response must start by"
                    f'"{self.func_call_token()} followed by a json like this: '
                    f'{{"call": "function_name", "params": {{"arg1": "value1"}}}}.\n'
                    "If you cannot call a function due to lack of information, "
                    "do not make a function call and ask the user for additional details."
                    f"If you can call a function, you don't explain anything, just do the call. "
                    f"You only call functions when it's needed and when the description matches with the user input. "
                    f"After a function call, the response must terminate immediately.\n\n"
                )
                if isinstance(request.messages, str):
                    request.messages = text_inject + request.messages
                elif isinstance(request.messages, list) and len(request.messages) >= 1:
                    request.messages[0].content = (
                        text_inject + request.messages[0].content
                    )


class PromptCapture:
    def __init__(self):
        self.content: str = ""
        self.ignore = False
        self.is_function_call = False
        self.calls_list: list[dict] = None

    def make_calls_list(self, prompter: OpenAIToolsPrompter):
        calls_list = self.content.split(prompter.func_call_token())
        self.calls_list = []
        for v_call in calls_list:
            if len(v_call):
                try:
                    call_dict = json.loads(v_call)
                    if "call" in call_dict:
                        self.calls_list.append(call_dict)
                except json.decoder.JSONDecodeError:
                    # Simply ignore invalid functions calls...
                    pass

    def num_calls(self):
        return len(self.calls_list) if self.calls_list is not None else 0

    def calls_validation(self, tools_list: [str]) -> bool:
        """Validate function / tool calls by searching name in the tools defined in the request."""
        if self.calls_list is not None:
            for ic in range(len(self.calls_list)):
                if self.calls_list[ic]["call"] in tools_list:
                    pass
                else:
                    return False
            return True
        return False

    def to_tool_call_message(self, func_id: int) -> Union[ToolCallsMessage, None]:
        if self.calls_list is not None:
            call = self.calls_list[func_id]
            arguments = call["params"] if "params" in call else None
            function_call = FunctionCall(
                name=call["call"], arguments=json.dumps(arguments)
            )
            return ToolCallsMessage(
                id="call_" + call["call"], type="function", function=function_call
            )
        return None

    def to_tool_call_delta(
        self, index: int, func_id: int
    ) -> Union[ToolCallsDelta, None]:
        mesg = self.to_tool_call_message(func_id)
        if mesg is not None:
            return ToolCallsDelta(
                index=index, id=mesg.id, type=mesg.type, function=mesg.function
            )
        return None
