from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult


class AgentCallBackHandler(BaseCallbackHandler):

    # This method is called for non-chat models (regular LLMs). If you're implementing a handler for a chat model,
    # you should use on_chat_model_start instead.
    def on_llm_start(
        self,
        serialized,
        prompts,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        **kwargs,
    ):
        print(f"Prompt to LLM was:***\n {prompts[0]}")
        print("***************")
        return super().on_llm_start(
            serialized,
            prompts,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        print(f"LLM Response:***\n {response.generations[0][0].text}")
        print("***********")
        return super().on_llm_end(
            response, run_id=run_id, parent_run_id=parent_run_id, **kwargs
        )
