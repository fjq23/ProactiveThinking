import copy
import json
import random
import re
from typing import Optional

from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.agent.llm_agent import (
    AGENT_INSTRUCTION as TAU2_AGENT_INSTRUCTION,
    SYSTEM_PROMPT as TAU2_SYSTEM_PROMPT,
)
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import generate
from tau2.utils.tools import parse_action_string

from helpers import (
    Tau2AgentState,
    attach_protocol_metadata,
    completion_tokens_from_message,
    compute_accepted_tokens,
    extract_content_logprobs,
    extract_prompt_logprobs_for_tokens,
    extract_prompt_logprobs_for_text,
    get_jaccard_similarity,
    normalize_method_name,
    render_history,
    tau2_branch_max_tokens_for_message,
)


THINK_PROMPT = """\
Reason step by step about the latest situation before acting.

Focus on:
- the user's latest need
- what has already been established
- whether a tool call is required
- what policy constraints apply

Keep the reasoning concise."""

TEXT_TOOL_PROMPT = """\
Decide the next action according to the policy and conversation history.

You must return exactly one JSON object:
- user reply: {{"message": "text to send to the user"}}
- tool call: {{"name": "tool_name", "arguments": {{"arg_name": "value"}}}}

Available tools:
{tools}

Do not include reasoning, markdown, code fences, XML tags, or any text outside the JSON object."""

REACTIVE_PROMPT = """\
Reason step by step about the latest situation and then decide the next action.

Focus on:
- the user's latest need
- what has already been established
- whether a tool call is required
- what policy constraints apply

Available tools:
{tools}

Return exactly this format:
<thought>
your concise reasoning
</thought>
<action>
either:
- user reply JSON: {{"message": "text to send to the user"}}
- tool call JSON: {{"name": "tool_name", "arguments": {{"arg_name": "value"}}}}
</action>

Do not use XML/function tags inside <action>; use only the JSON formats above.
Do not include any text outside these tags."""

SPECULATE_REPLY_PROMPT = """\
You are planning ahead for a customer service conversation.

Conversation history before the user's latest reply:
{history}

List the 2 most likely possible latest user replies, sorted by likelihood.
Return exactly this format:
<response> ... </response>
<response> ... </response>"""

PREVIOUS_PROACTIVE_PLAN_PROMPT = """\
You are planning one step ahead for a customer service conversation.

Conversation history before the latest user reply:
{history}

Hypothetical latest user reply:
{user_reply}

Briefly prepare the best immediate next move. The move may be either:
- a short reply to the user
- a tool call if a tool is needed

Return a short plain-text plan that states:
- whether to respond or call a tool
- what the next move should focus on
- any key policy consideration"""

PREVIOUS_PROACTIVE_FINAL_PROMPT = """\
Conversation history before the latest user reply:
{history}

Speculative plans prepared before the latest user reply:
{thinking}

Actual latest user reply:
{response}

Use the actual latest user reply as the newest turn in the conversation.
The speculative plans may help, but they are not binding. Revise them freely if the actual reply points elsewhere.
Now take the best next action according to the policy."""

PREVIOUS_PROACTIVE_FINAL_PROMPT_WITH_THINK = """\
Conversation history before the latest user reply:
{history}

Speculative plans prepared before the latest user reply:
{thinking}

Actual latest user reply:
{response}

Use the actual latest user reply as the newest turn in the conversation.
The speculative plans may help, but they are not binding. Revise them freely if the actual reply points elsewhere.

Reason step by step about the latest situation and then decide the next action.

Focus on:
- the user's latest need
- what has already been established
- whether a tool call is required
- what policy constraints apply

Available tools:
{tools}

Return exactly this format:
<thought>
your concise reasoning
</thought>
<action>
either:
- user reply JSON: {{"message": "text to send to the user"}}
- tool call JSON: {{"name": "tool_name", "arguments": {{"arg_name": "value"}}}}
</action>

Do not use XML/function tags inside <action>; use only the JSON formats above.
Do not include any text outside these tags."""

class BenchmarkTau2Agent(HalfDuplexAgent[Tau2AgentState]):
    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[dict] = None,
        method: str = "basic",
        gamma: float = 1.0,
        spec_cot: bool = False,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.llm = llm
        self.llm_args = dict(llm_args or {})
        method_from_args = self.llm_args.pop("method", None)
        gamma_from_args = self.llm_args.pop("gamma", None)
        spec_cot_from_args = self.llm_args.pop("spec_cot", None)
        self.method = normalize_method_name(method)
        if method_from_args:
            self.method = normalize_method_name(str(method_from_args))
        if gamma_from_args is not None:
            gamma = gamma_from_args
        if spec_cot_from_args is not None:
            spec_cot = spec_cot_from_args
        self.gamma = float(gamma)
        self.spec_cot = bool(spec_cot)
        self._rng = random.Random()
        self._tool_prompt = self._render_tool_prompt()

    @property
    def system_prompt(self) -> str:
        return TAU2_SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy,
            agent_instruction=TAU2_AGENT_INSTRUCTION,
        )

    def set_seed(self, seed: int):
        self._rng.seed(seed)
        super().set_seed(seed)

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> Tau2AgentState:
        return Tau2AgentState(
            system_messages=[
                SystemMessage(role="system", content=self.system_prompt)
            ],
            messages=list(message_history) if message_history else [],
        )

    @staticmethod
    def _merge_nested_dicts(base: Optional[dict], override: Optional[dict]) -> dict:
        merged = dict(base or {})
        for key, value in (override or {}).items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                nested = dict(merged[key])
                nested.update(value)
                merged[key] = nested
            else:
                merged[key] = value
        return merged

    def _llm_args(self, extra_llm_args: Optional[dict] = None) -> dict:
        llm_args = dict(self.llm_args)
        extra_body = dict(llm_args.pop("extra_body", {}) or {})

        if extra_llm_args:
            extra_llm_args = dict(extra_llm_args)
            extra_body = self._merge_nested_dicts(
                extra_body, extra_llm_args.pop("extra_body", None)
            )
            llm_args.update(extra_llm_args)

        chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
        chat_template_kwargs.setdefault("enable_thinking", False)
        extra_body["chat_template_kwargs"] = chat_template_kwargs
        llm_args["extra_body"] = extra_body
        return llm_args

    def _proactive_branch_llm_args(self, latest_message: Message) -> dict:
        return {"max_tokens": tau2_branch_max_tokens_for_message(latest_message)}

    @staticmethod
    def _is_continue_final_message_error(exc: Exception) -> bool:
        message = str(exc)
        return "continue_final_message" in message and (
            "final message does not appear in the chat after applying the chat template"
            in message
            or "final message has no content to continue" in message
            or "could not find any text to continue in the final message" in message
        )

    def _think(self, state: Tau2AgentState) -> str:
        response = generate(
            model=self.llm,
            tools=[],
            messages=state.system_messages
            + state.messages
            + [UserMessage(role="user", content=THINK_PROMPT)],
            call_name="tau2_reactive_think",
            **self._llm_args(),
        )
        return str(response.content or "")

    def _render_tool_prompt(self) -> str:
        tool_schemas = [tool.openai_schema for tool in self.tools]
        return json.dumps(tool_schemas, indent=2, ensure_ascii=False)

    def _basic_text_tool_prompt(self) -> str:
        return TEXT_TOOL_PROMPT.format(tools=self._tool_prompt)

    def _reactive_text_tool_prompt(self) -> str:
        return REACTIVE_PROMPT.format(tools=self._tool_prompt)

    @staticmethod
    def _split_reactive_output(content: str) -> tuple[str, Optional[str], str]:
        thought_parts = [
            match.group(2).strip()
            for match in re.finditer(r"<(thought|think)>(.*?)</\1>", content, re.S)
            if match.group(2).strip()
        ]
        thought_text = "\n\n".join(thought_parts)

        action_match = re.search(
            r"<action>(.*?)(?:</action>|</tool_call>|$)", content, re.S
        )
        if action_match:
            action_text = action_match.group(1).strip()
        else:
            tool_call_match = re.search(
                r"<tool_call>(.*?)(?:</tool_call>|</action>|$)", content, re.S
            )
            action_text = tool_call_match.group(1).strip() if tool_call_match else None

        visible_content = re.sub(r"<(thought|think)>.*?</\1>", "", content, flags=re.S)
        if action_match:
            visible_content = visible_content.replace(action_match.group(0), "", 1)
        else:
            visible_content = re.sub(
                r"<tool_call>.*?(?:</tool_call>|</action>|$)",
                "",
                visible_content,
                flags=re.S,
            )
        visible_content = visible_content.strip()
        return thought_text, action_text, visible_content

    @staticmethod
    def _normalize_action_text(action_text: str) -> str:
        action_text = action_text.strip()
        fence_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", action_text, re.S)
        if fence_match:
            action_text = fence_match.group(1).strip()

        xml_tool_call = BenchmarkTau2Agent._normalize_xml_tool_call(action_text)
        if xml_tool_call is not None:
            return xml_tool_call

        try:
            action_json = json.loads(action_text)
        except json.JSONDecodeError:
            return action_text

        if not isinstance(action_json, dict):
            return action_text

        def message_from_named_action(name, arguments):
            if not isinstance(name, str):
                return None
            normalized_name = re.sub(r"[^a-z0-9]+", "", name.lower())
            if normalized_name not in {
                "message",
                "reply",
                "response",
                "respond",
                "userreply",
                "assistantreply",
                "respondtouser",
                "speak",
                "say",
            }:
                return None
            if isinstance(arguments, str) and arguments.strip():
                return arguments.strip()
            arguments = normalize_arguments(arguments)
            if not isinstance(arguments, dict):
                return None
            for key in ("message", "content", "response", "reply", "text"):
                value = arguments.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return None

        def normalize_arguments(arguments):
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    return None
            return arguments if isinstance(arguments, dict) else None

        def normalized_tool_call(name, arguments):
            message = message_from_named_action(name, arguments)
            if message is not None:
                return message
            arguments = normalize_arguments(arguments)
            if isinstance(name, str) and arguments is not None:
                return json.dumps(
                    {
                        "name": name,
                        "arguments": arguments,
                    }
                )
            return None

        if isinstance(action_json.get("tool_calls"), list) and action_json["tool_calls"]:
            tool_call = action_json["tool_calls"][0]
            if isinstance(tool_call, dict):
                function = tool_call.get("function")
                if isinstance(function, dict):
                    normalized = normalized_tool_call(
                        function.get("name"), function.get("arguments")
                    )
                    if normalized is not None:
                        return normalized

        for wrapper_key in ("function_call", "tool_call", "action"):
            wrapper = action_json.get(wrapper_key)
            if isinstance(wrapper, dict):
                arguments = (
                    wrapper.get("arguments")
                    if "arguments" in wrapper
                    else wrapper.get("parameters", wrapper.get("parameter"))
                )
                normalized = normalized_tool_call(wrapper.get("name"), arguments)
                if normalized is not None:
                    return normalized
                for key in ("message", "content", "response", "reply"):
                    value = wrapper.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
            elif isinstance(wrapper, str) and wrapper.strip():
                return BenchmarkTau2Agent._normalize_action_text(wrapper)

        arguments = (
            action_json.get("arguments")
            if "arguments" in action_json
            else action_json.get("parameters", action_json.get("parameter"))
        )
        normalized = normalized_tool_call(
            action_json.get("name"), arguments
        )
        if normalized is not None:
            return normalized

        if "function" in action_json:
            function = action_json["function"]
            if isinstance(function, dict):
                normalized = normalized_tool_call(
                    function.get("name"),
                    function.get("arguments", action_json.get("arguments")),
                )
            else:
                arguments = (
                    action_json.get("parameters")
                    if "parameters" in action_json
                    else action_json.get("parameter", action_json.get("arguments"))
                )
                normalized = normalized_tool_call(function, arguments)
            if normalized is not None:
                return normalized

        for key in ("message", "content", "response", "reply"):
            value = action_json.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        return action_text

    @staticmethod
    def _normalize_xml_tool_call(action_text: str) -> Optional[str]:
        action_text = action_text.strip()
        tool_call_match = re.fullmatch(
            r"<tool_call>\s*(.*?)\s*</tool_call>", action_text, re.S
        )
        if tool_call_match:
            action_text = tool_call_match.group(1).strip()

        function_match = re.search(
            r"<function(?:=|>|:)([^>\n<]+)>?\s*(.*?)</function>", action_text, re.S
        )
        if not function_match:
            function_match = re.search(
                r"<function\s+name=[\"']([^\"']+)[\"']>\s*(.*?)</function>",
                action_text,
                re.S,
            )
        if not function_match:
            return None

        function_name = function_match.group(1).strip()
        body = function_match.group(2)
        arguments = {}
        for parameter_name, parameter_value in re.findall(
            r"<parameter(?:=|>)([^>\n<]+)>?\s*(.*?)</parameter>", body, re.S
        ):
            arguments[parameter_name.strip()] = parameter_value.strip()
        for parameter_name, parameter_value in re.findall(
            r"<parameter\s+name=[\"']([^\"']+)[\"']>\s*(.*?)</parameter>", body, re.S
        ):
            arguments[parameter_name.strip()] = parameter_value.strip()

        if not function_name:
            return None
        return json.dumps({"name": function_name, "arguments": arguments})

    def _attach_reactive_metadata(
        self,
        message: AssistantMessage,
        *,
        full_response: str,
        thought: str,
        action_text: Optional[str],
    ) -> AssistantMessage:
        raw_data = dict(message.raw_data or {})
        raw_data["benchmark_tau2_reactive"] = {
            "full_response": full_response,
            "thought": thought,
            "action_text": action_text,
        }
        message.raw_data = raw_data
        return message

    def _parse_action_output(
        self, response: AssistantMessage, *, record_reactive: bool = True
    ) -> AssistantMessage:
        content = response.content or ""
        thought, action_text, visible_content = self._split_reactive_output(content)
        action_text = action_text if action_text is not None else visible_content
        if not action_text:
            raise ValueError("Reactive response did not contain an action.")

        action_text = self._normalize_action_text(action_text)
        parsed_message = parse_action_string(action_text, requestor="assistant")
        parsed_message.cost = response.cost
        parsed_message.usage = response.usage
        raw_data = dict(response.raw_data or {})
        raw_data["benchmark_tau2_text_tool"] = {
            "full_response": content,
            "action_text": action_text,
        }
        if record_reactive:
            raw_data["benchmark_tau2_reactive"] = {
                "full_response": content,
                "thought": thought,
                "action_text": action_text,
            }
        parsed_message.raw_data = raw_data
        parsed_message.generation_time_seconds = response.generation_time_seconds
        return parsed_message

    def _generate_text_tool_action(
        self,
        messages: list[Message],
        *,
        call_name: str,
        record_reactive: bool = False,
        extra_instruction: Optional[str] = None,
        extra_llm_args: Optional[dict] = None,
    ) -> tuple[AssistantMessage, AssistantMessage]:
        prompt = self._basic_text_tool_prompt()
        if extra_instruction:
            prompt = f"{extra_instruction}\n\n{prompt}"
        llm_args = self._llm_args(extra_llm_args)
        response = generate(
            model=self.llm,
            tools=[],
            messages=messages + [UserMessage(role="user", content=prompt)],
            call_name=call_name,
            **llm_args,
        )
        try:
            assistant_message = self._parse_action_output(
                response, record_reactive=record_reactive
            )
        except ValueError:
            assistant_message = response
        return response, assistant_message

    def _attach_delay_from_response(
        self,
        assistant_message: AssistantMessage,
        response: AssistantMessage,
        *,
        middle_delay_tokens: int = 0,
        accepted_count: Optional[int] = None,
        speculative_used: bool = False,
        selected_reply: Optional[str] = None,
    ) -> AssistantMessage:
        return attach_protocol_metadata(
            assistant_message,
            delay_tokens=completion_tokens_from_message(response),
            middle_delay_tokens=middle_delay_tokens,
            accepted_count=accepted_count,
            speculative_used=speculative_used,
            selected_reply=selected_reply,
        )

    def _act(
        self,
        state: Tau2AgentState,
        instruction: Optional[str] = None,
        extra_llm_args: Optional[dict] = None,
    ) -> AssistantMessage:
        messages = state.system_messages + state.messages
        if instruction:
            messages = messages + [UserMessage(role="user", content=instruction)]
        llm_args = self._llm_args(extra_llm_args)
        return generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            call_name="tau2_agent_act",
            **llm_args,
        )

    def _generate_basic(self, state: Tau2AgentState) -> AssistantMessage:
        response, assistant_message = self._generate_text_tool_action(
            state.system_messages + state.messages,
            call_name="tau2_basic_text_tool_act",
            record_reactive=False,
        )
        return self._attach_delay_from_response(assistant_message, response)

    def _generate_reactive(self, state: Tau2AgentState) -> AssistantMessage:
        response = generate(
            model=self.llm,
            tools=[],
            messages=state.system_messages
            + state.messages
            + [UserMessage(role="user", content=self._reactive_text_tool_prompt())],
            call_name="tau2_reactive_act",
            **self._llm_args(),
        )
        if response.tool_calls:
            content = response.content or ""
            thought, action_text, visible_content = self._split_reactive_output(content)
            response.content = visible_content
            self._attach_reactive_metadata(
                response,
                full_response=content,
                thought=thought,
                action_text=action_text,
            )
            assistant_message = response
        else:
            try:
                assistant_message = self._parse_action_output(response)
            except ValueError:
                return self._generate_basic(state)
        return self._attach_delay_from_response(assistant_message, response)

    @staticmethod
    def _render_previous_proactive_plan(message: AssistantMessage) -> str:
        if message.tool_calls:
            action = "; ".join(
                f"{tool_call.name}({tool_call.arguments})"
                for tool_call in message.tool_calls
            )
        else:
            action = (message.content or "").strip()

        metadata = (message.raw_data or {}).get("benchmark_tau2_reactive") or {}
        thought = (metadata.get("thought") or "").strip()
        if thought and action:
            return f"Thought: {thought}\nAction: {action}"
        return action or thought

    @staticmethod
    def _raw_generation_text(message: AssistantMessage) -> str:
        raw_data = message.raw_data or {}
        for key in ("benchmark_tau2_reactive", "benchmark_tau2_text_tool"):
            metadata = raw_data.get(key)
            if isinstance(metadata, dict):
                full_response = metadata.get("full_response")
                if isinstance(full_response, str) and full_response.strip():
                    return full_response
        return message.content or ""

    def _speculate_replies(self, history_messages: list[Message]) -> list[str]:
        prompt = SPECULATE_REPLY_PROMPT.format(history=render_history(history_messages))
        if self.spec_cot:
            prompt += "\n\nBriefly think first, then still return only the required <response> tags."

        response = generate(
            model=self.llm,
            tools=[],
            messages=[
                SystemMessage(role="system", content=self.system_prompt),
                UserMessage(role="user", content=prompt),
            ],
            call_name="tau2_speculate_replies",
            **self._llm_args(),
        )
        content = response.content or ""
        replies = [
            item.strip()
            for item in re.findall(r"<response>(.*?)</response>", content, re.S)
            if item.strip()
        ]
        if not replies and content.strip():
            replies = [content.strip()]
        return replies[:2]

    def _generate_previous_proactive(self, state: Tau2AgentState) -> AssistantMessage:
        latest_message = state.messages[-1]
        history_messages = state.messages[:-1]
        speculative_replies = self._speculate_replies(history_messages)

        analyses = []
        branch_token_counts: list[int] = []
        for reply in speculative_replies[:1]:
            speculative_messages = history_messages + [
                UserMessage(role="user", content=reply)
            ]
            plan_response = generate(
                model=self.llm,
                tools=[],
                messages=state.system_messages
                + speculative_messages
                + [UserMessage(role="user", content=self._reactive_text_tool_prompt())],
                call_name="tau2_previous_proactive_prepare",
                **self._llm_args(self._proactive_branch_llm_args(latest_message)),
            )
            branch_token_counts.append(completion_tokens_from_message(plan_response))
            try:
                plan_message = self._parse_action_output(plan_response)
                plan = self._render_previous_proactive_plan(plan_message)
            except ValueError:
                plan = (plan_response.content or "").strip()
            analyses.append(
                f'- If the user responds, "{reply}", you may then respond, "{plan}".'
            )
        analyses.append(
            "- You may ask other questions if the user's actual response was not included in the analysis."
        )
        analysis_string = "\n".join(analyses)

        response, assistant_message = self._generate_text_tool_action(
            [
                SystemMessage(role="system", content=self.system_prompt),
                UserMessage(
                    role="user",
                    content=PREVIOUS_PROACTIVE_FINAL_PROMPT.format(
                        history=render_history(history_messages),
                        thinking=analysis_string,
                        response=latest_message.content or "",
                    ),
                ),
            ],
            call_name="tau2_previous_proactive_final",
            record_reactive=False,
        )
        selected_reply = speculative_replies[0] if speculative_replies else None
        return self._attach_delay_from_response(
            assistant_message,
            response,
            middle_delay_tokens=max(branch_token_counts, default=0),
            selected_reply=selected_reply,
        )

    def _generate_previous_proactive_with_think(
        self, state: Tau2AgentState
    ) -> AssistantMessage:
        latest_message = state.messages[-1]
        history_messages = state.messages[:-1]
        speculative_replies = self._speculate_replies(history_messages)

        analyses = []
        branch_token_counts: list[int] = []
        for reply in speculative_replies[:1]:
            speculative_messages = history_messages + [
                UserMessage(role="user", content=reply)
            ]
            plan_response = generate(
                model=self.llm,
                tools=[],
                messages=state.system_messages
                + speculative_messages
                + [UserMessage(role="user", content=self._reactive_text_tool_prompt())],
                call_name="tau2_previous_proactive_with_think_prepare",
                **self._llm_args(self._proactive_branch_llm_args(latest_message)),
            )
            branch_token_counts.append(completion_tokens_from_message(plan_response))
            try:
                plan_message = self._parse_action_output(plan_response)
                plan = self._render_previous_proactive_plan(plan_message)
            except ValueError:
                plan = (plan_response.content or "").strip()
            analyses.append(
                f'- If the user responds, "{reply}", you may then respond, "{plan}".'
            )
        analyses.append(
            "- You may ask other questions if the user's actual response was not included in the analysis."
        )
        analysis_string = "\n".join(analyses)

        response = generate(
            model=self.llm,
            tools=[],
            messages=state.system_messages
            + [
                UserMessage(
                    role="user",
                    content=PREVIOUS_PROACTIVE_FINAL_PROMPT_WITH_THINK.format(
                        history=render_history(history_messages),
                        thinking=analysis_string,
                        response=latest_message.content or "",
                        tools=self._tool_prompt,
                    ),
                )
            ],
            call_name="tau2_previous_proactive_with_think_final",
            **self._llm_args(),
        )
        try:
            assistant_message = self._parse_action_output(response)
        except ValueError:
            return self._generate_basic(state)
        selected_reply = speculative_replies[0] if speculative_replies else None
        return self._attach_delay_from_response(
            assistant_message,
            response,
            middle_delay_tokens=max(branch_token_counts, default=0),
            selected_reply=selected_reply,
        )

    def _generate_speculative_draft(
        self,
        history_messages: list[Message],
        speculative_reply: str,
        latest_message: Message,
    ) -> AssistantMessage:
        speculative_state = Tau2AgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=history_messages + [UserMessage(role="user", content=speculative_reply)],
        )
        response = generate(
            model=self.llm,
            tools=[],
            messages=speculative_state.system_messages
            + speculative_state.messages
            + [UserMessage(role="user", content=self._reactive_text_tool_prompt())],
            call_name="tau2_speculative_draft",
            **self._llm_args(
                {
                    "logprobs": True,
                    **self._proactive_branch_llm_args(latest_message),
                }
            ),
        )
        try:
            return self._parse_action_output(response)
        except ValueError:
            return response

    def _continue_from_prefill(
        self, state: Tau2AgentState, prefill: str
    ) -> tuple[AssistantMessage, AssistantMessage]:
        if not prefill.strip():
            raise ValueError("Cannot continue from an empty assistant prefill")

        try:
            response = generate(
                model=self.llm,
                tools=[],
                messages=state.system_messages
                + state.messages
                + [
                    UserMessage(role="user", content=self._reactive_text_tool_prompt()),
                    AssistantMessage(role="assistant", content=prefill),
                ],
                call_name="tau2_speculative_continue",
                **self._llm_args(
                    {
                        "extra_body": {
                            "continue_final_message": True,
                            "add_generation_prompt": False,
                        }
                    }
                ),
            )
        except Exception as exc:
            if self._is_continue_final_message_error(exc):
                raise ValueError("Speculative continuation is incompatible with chat template") from exc
            raise
        parse_response = copy.copy(response)
        parse_response.content = f"{prefill}{response.content or ''}"
        try:
            assistant_message = self._parse_action_output(parse_response)
        except ValueError:
            assistant_message = parse_response
        return response, assistant_message

    def _generate_speculative_proactive(self, state: Tau2AgentState) -> AssistantMessage:
        latest_message = state.messages[-1]

        history_messages = state.messages[:-1]
        speculative_replies = self._speculate_replies(history_messages)
        if not speculative_replies:
            return self._generate_reactive(state)

        speculative_actions: list[tuple[str, AssistantMessage, int]] = []
        for reply in speculative_replies:
            try:
                draft = self._generate_speculative_draft(
                    history_messages, reply, latest_message
                )
            except ValueError:
                continue
            if not (draft.has_content() or draft.is_tool_call()):
                continue
            speculative_actions.append(
                (reply, draft, completion_tokens_from_message(draft))
            )

        if not speculative_actions:
            return self._generate_reactive(state)

        best_reply, best_action, _ = max(
            speculative_actions,
            key=lambda item: get_jaccard_similarity(latest_message.content or "", item[0]),
        )
        middle_delay_tokens = max((item[2] for item in speculative_actions), default=0)

        verification_text = self._raw_generation_text(best_action)
        draft_logprobs = extract_content_logprobs(best_action)
        if draft_logprobs is None:
            return self._generate_reactive(state)
        if not verification_text:
            return self._generate_reactive(state)
        try:
            verify_message = generate(
                model=self.llm,
                tools=[],
                messages=state.system_messages
                + state.messages
                + [
                    UserMessage(role="user", content=self._reactive_text_tool_prompt()),
                    AssistantMessage(role="assistant", content=verification_text),
                ],
                call_name="tau2_speculative_verify",
                max_tokens=1,
                **self._llm_args(
                    {
                        "extra_body": {
                            "prompt_logprobs": 1,
                            "add_generation_prompt": False,
                            "continue_final_message": True,
                        }
                    }
                ),
            )
        except Exception as exc:
            if self._is_continue_final_message_error(exc):
                return self._generate_reactive(state)
            raise
        draft_tokens, draft_token_logprobs = draft_logprobs
        ref_logprobs = extract_prompt_logprobs_for_tokens(verify_message, draft_tokens)
        if ref_logprobs is None:
            ref_logprobs = extract_prompt_logprobs_for_text(
                verify_message, verification_text
            )
        if ref_logprobs is None or len(ref_logprobs) != len(draft_token_logprobs):
            return self._generate_reactive(state)

        accepted_count = compute_accepted_tokens(
            draft_token_logprobs,
            ref_logprobs,
            gamma=self.gamma,
            rng=self._rng,
        )

        prefill = "".join(draft_tokens[:accepted_count])
        if not prefill.strip():
            return self._generate_reactive(state)

        try:
            response, continuation = self._continue_from_prefill(state, prefill)
        except ValueError as exc:
            if "Speculative continuation is incompatible with chat template" in str(exc):
                return self._generate_reactive(state)
            raise

        return self._attach_delay_from_response(
            continuation,
            response,
            middle_delay_tokens=middle_delay_tokens,
            accepted_count=accepted_count,
            speculative_used=True,
            selected_reply=best_reply,
        )

    def _ensure_nonempty_message(
        self, state: Tau2AgentState, assistant_message: AssistantMessage
    ) -> AssistantMessage:
        if assistant_message.has_content() or assistant_message.is_tool_call():
            return assistant_message

        fallback_message = self._generate_basic(state)
        if fallback_message.has_content() or fallback_message.is_tool_call():
            return fallback_message

        raise ValueError(
            f"Agent produced an empty assistant message in method={self.method}"
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: Tau2AgentState
    ) -> tuple[AssistantMessage, Tau2AgentState]:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        if self.method in {"basic", "base", "no_cot"}:
            assistant_message = self._generate_basic(state)
        elif self.method == "reactive":
            assistant_message = self._generate_reactive(state)
        elif self.method == "previous_proactive":
            assistant_message = self._generate_previous_proactive(state)
        elif self.method == "previous_proactive_with_think":
            assistant_message = self._generate_previous_proactive_with_think(state)
        else:
            assistant_message = self._generate_speculative_proactive(state)

        assistant_message = self._ensure_nonempty_message(state, assistant_message)
        state.messages.append(assistant_message)
        return assistant_message, state


def create_benchmark_tau2_agent(tools, domain_policy, **kwargs):
    llm_args = dict(kwargs.get("llm_args") or {})
    return BenchmarkTau2Agent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=llm_args,
        method=kwargs.get("method", llm_args.get("method", "basic")),
        gamma=kwargs.get("gamma", llm_args.get("gamma", 1.0)),
        spec_cot=kwargs.get("spec_cot", llm_args.get("spec_cot", False)),
    )
