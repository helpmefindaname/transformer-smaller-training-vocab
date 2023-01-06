name: Bug Report
description: File a bug report
title: "[Bug]: "
labels: ["bug"]
assignees: []
body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time to fill out this bug report!
  - type: textarea
    id: what-happened
    attributes:
      label: What happened?
      description: Also tell us, what did you expect to happen?
      placeholder: Tell us what you see!
      value: "A bug happened!"
    validations:
      required: true
- type: textarea
  attributes:
    label: Environment
    description: |
      examples:
        - **transformer-smaller-training-vocab**: current main
        - **torch**: 1.5.0
        - **transformers**: 4.5.0
    value: |
        - transformer-smaller-training-vocab:
        - torch:
        - transformers:
  - type: textarea
    id: logs
    attributes:
      label: Relevant log output
      description: Please copy and paste any relevant log output. This will be automatically formatted into code, so no need for backticks.
      render: shell