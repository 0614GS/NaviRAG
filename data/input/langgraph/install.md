# Install LangGraph

To install the base LangGraph package:

<CodeGroup>
  ```bash pip theme={null}
  pip install -U langgraph
  ```

  ```bash uv theme={null}
  uv add langgraph
  ```
</CodeGroup>

To use LangGraph you will usually want to access LLMs and define tools.
You can do this however you see fit.

One way to do this (which we will use in the docs) is to use [LangChain](/oss/python/langchain/overview).

Install LangChain with:

<CodeGroup>
  ```bash pip theme={null}
  pip install -U langchain
  # Requires Python 3.10+
  ```

  ```bash uv theme={null}
  uv add langchain
  # Requires Python 3.10+
  ```
</CodeGroup>

To work with specific LLM provider packages, you will need install them separately.

Refer to the [integrations](/oss/python/integrations/providers/overview) page for provider-specific installation instructions.

***

<Callout icon="pen-to-square" iconType="regular">
  [Edit this page on GitHub](https://github.com/langchain-ai/docs/edit/main/src/oss/langgraph/install.mdx) or [file an issue](https://github.com/langchain-ai/docs/issues/new/choose).
</Callout>

<Tip icon="terminal" iconType="regular">
  [Connect these docs](/use-these-docs) to Claude, VSCode, and more via MCP for real-time answers.
</Tip>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.langchain.com/llms.txt