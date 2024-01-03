import json
from typing import Union
from vllm.logger import init_logger
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolParam,
    ChoiceDeltaToolCall,
    ChatCompletionMessageToolCall,
    Function,
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
                if isinstance(request.messages, str):
                    current_request = request.messages
                elif isinstance(request.messages, list) and len(request.messages) >= 1:
                    current_request = request.messages[0].content
                text_inject = "The question to be answered is the following:\n"
                text_inject += current_request + "\n"
                text_inject += (
                    "You are given data repositories to help you answer the question.\n"
                )
                text_inject += (
                    "The data repositories are given in the following format:\n"
                )
                text_inject += '    "repository_name": "A short description of the data that is in the specific repository."\n'
                text_inject += "The repositories available are the following:\n"
                text_inject += "\n\n"
                for tool in tools_list:
                    if tool.type == "function":
                        text_inject += f'    "{tool.function.name}": "{tool.function.description}"\n'
                text_inject += "\n\n"
                text_inject += (
                    f"You can either answer directly or select one or multiple of the repositories above.\n"
                    f"You need to call a at least one data repository you can either forward the question to be answered or reformulate to fit it better to the data repository.\n"
                    f'To call a data repository, the message must start by "{self.func_call_token()}" followed by the name and the question like this:\n'
                    f'  {self.func_call_token()}"repository_name": "A more specific question based on the question to be answered?"\n'
                    f"End of data repository instructions.\n"
                    f"Be sure to call at least one data repository.\n"
                    f"Be very short and concise in your answer. An answer template is like this: \n"
                    "=========\n"
                    f"I would call the following data repositories with the following questions:\n"
                    f'  {self.func_call_token()}"repository_name": "A more specific question based on the question to be answered?"\n'
                    "=========\n"
                    f"Do not write anything else than the answer template above.\n"
                )
                if isinstance(request.messages, str):
                    request.messages = text_inject
                elif isinstance(request.messages, list) and len(request.messages) >= 1:
                    request.messages[0].content = text_inject


class PromptCapture:
    def __init__(self):
        self.content: str = ""
        self.maybe_function_call = False
        self.is_function_call = False
        self.prefix_size = 0
        self.calls_list: list[dict] = []

    def reset(self, reset_calls_list=False):
        self.content = ""
        self.maybe_function_call = False
        self.is_function_call = False
        self.prefix_size = 0
        if reset_calls_list:
            self.calls_list = []

    def num_calls(self):
        return len(self.calls_list)

    def make_calls_list(self, prompter: OpenAIToolsPrompter):
        calls_list = self.content.split(prompter.func_call_token())
        for v_call in calls_list:
            if len(v_call):
                try:
                    call_dict = json.loads(v_call)
                    if "call" in call_dict:
                        self.calls_list.append(call_dict)
                except json.decoder.JSONDecodeError:
                    # Simply ignore invalid functions calls...
                    pass

    def validate_call(self, call_id: int, tools_list: [str]) -> int:
        """Validate function / tool calls by searching name in the tools defined in the request.
        Returns the function id or -1 on failure."""
        if len(self.calls_list) and call_id < len(self.calls_list):
            try:
                return tools_list.index(self.calls_list[call_id]["call"])
            except ValueError:
                pass
        return -1

    def to_tool_call_message(
        self, call_id: int
    ) -> Union[ChatCompletionMessageToolCall, None]:
        if self.calls_list is not None:
            call = self.calls_list[call_id]
            arguments = call["params"] if "params" in call else None
            function_call = Function(name=call["call"], arguments=json.dumps(arguments))
            return ChatCompletionMessageToolCall(
                id="call_" + call["call"], type="function", function=function_call
            )
        return None

    def to_tool_call_delta(
        self, index: int, call_id: int
    ) -> Union[ChoiceDeltaToolCall, None]:
        mesg = self.to_tool_call_message(call_id)
        if mesg is not None:
            return ChoiceDeltaToolCall(
                index=index, id=mesg.id, type=mesg.type, function=mesg.function
            )
        return None
