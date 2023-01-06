name: Add tokenizer support
description: Add support for a slow tokenizer
title: "[TOKENIZER-REQUEST]: "
labels: ["TOKENIZER-REQUEST"]
assignees:
body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time to fill out this tokenizer support form!
  - type: textarea
    id: tokenizer-name
    attributes:
      label: Tokenizer Name
      description: Please link a model that uses the tokenizer you want to be supported.
      placeholder: https://huggingface.co/<tokenizer-name>
      value: "https://huggingface.co/"
    validations:
      required: false
  - type: textarea
    id: context
    attributes:
      label: Additional Context
      description: Please tell us any context that is relevant!
      placeholder: I am using dataset xy.
      value: ""
    validations:
      required: false
  - type: textarea
    id: logs
    attributes:
      label: Relevant log output
      description: Please copy and paste any relevant log output. This will be automatically formatted into code, so no need for backticks.
      render: shell
---
name: Add tokenizer support
about: Describe what tokenizer you want to be supported
title: "[TOKENIZER-REQUEST]: "
labels: 'TOKENIZER-REQUEST'
assignees: ''

---

Hello I would like to have support for the slow tokenizer of [<tokenizer-name>](https://huggingface.co/<tokenizer-name>)
