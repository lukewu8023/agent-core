from datetime import datetime
from typing import Any
from uuid import uuid4
import logging

from agent_core.protocols.a2a.types import (
    Artifact,
    MessageSendParams,
    Message,
    Part,
    Role,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)


logger = logging.getLogger(__name__)


def create_task_obj(message_send_params: MessageSendParams) -> Task:
    """Create a new task object from message send params."""
    if not message_send_params.message.contextId:
        message_send_params.message.contextId = str(uuid4())

    return Task(
        id=str(uuid4()),
        contextId=message_send_params.message.contextId,
        status=TaskStatus(state=TaskState.submitted),
        history=[message_send_params.message],
    )


def append_artifact_to_task(task: Task, event: TaskArtifactUpdateEvent) -> None:
    """Helper method for updating Task with new artifact data."""
    if not task.artifacts:
        task.artifacts = []

    new_artifact_data: Artifact = event.artifact
    artifact_id: str = new_artifact_data.artifactId
    append_parts: bool = event.append or False

    existing_artifact: Artifact | None = None
    existing_artifact_list_index: int | None = None

    # Find existing artifact by its id
    for i, art in enumerate(task.artifacts):
        if hasattr(art, 'artifactId') and art.artifactId == artifact_id:
            existing_artifact = art
            existing_artifact_list_index = i
            break

    if not append_parts:
        # This represents the first chunk for this artifact index.
        if existing_artifact_list_index is not None:
            # Replace the existing artifact entirely with the new data
            logger.debug(
                f'Replacing artifact at id {artifact_id} for task {task.id}'
            )
            task.artifacts[existing_artifact_list_index] = new_artifact_data
        else:
            # Append the new artifact since no artifact with this index exists yet
            logger.debug(
                f'Adding new artifact with id {artifact_id} for task {task.id}'
            )
            task.artifacts.append(new_artifact_data)
    elif existing_artifact:
        # Append new parts to the existing artifact's parts list
        logger.debug(
            f'Appending parts to artifact id {artifact_id} for task {task.id}'
        )
        existing_artifact.parts.extend(new_artifact_data.parts)
    else:
        # We received a chunk to append, but we don't have an existing artifact.
        # we will ignore this chunk
        logger.warning(
            f'Received append=True for nonexistent artifact index {artifact_id} in task {task.id}. Ignoring chunk.'
        )


def update_task_with_agent_reasoning(
    task: Task, agent_reasoning: str
) -> None:
    """Updates the provided task with the agent response."""
    task.status.timestamp = datetime.now().isoformat()
    parts: list[Part] = []
    for reasoning in agent_reasoning:
        parts.append(Part(TextPart(text=reasoning)))
    task.status.state = TaskState.working
    task.status.message = None
    if not task.artifacts:
        task.artifacts = []

    artifact: Artifact = Artifact(parts=parts, artifactId=str(uuid4()))
    task.artifacts.append(artifact)


def update_task_with_agent_response(
    task: Task, agent_response: str
) -> None:
    """Updates the provided task with the agent response."""
    task.status.timestamp = datetime.now().isoformat()
    parts: list[Part] = [Part(TextPart(text=agent_response))]
    # if agent_response['require_user_input']:
    #     task.status.state = TaskState.input_required
    #     message = Message(
    #         messageId=str(uuid4()),
    #         role=Role.agent,
    #         parts=parts,
    #     )
    #     task.status.message = message
    #     if not task.history:
    #         task.history = []
    #
    #     task.history.append(message)
    # else:
    task.status.state = TaskState.completed
    task.status.message = None
    if not task.artifacts:
        task.artifacts = []

    artifact: Artifact = Artifact(parts=parts, artifactId=str(uuid4()))
    task.artifacts.append(artifact)


def process_streaming_agent_response(
    task: Task,
    agent_response: dict[str, Any],
) -> tuple[TaskArtifactUpdateEvent | None, TaskStatusUpdateEvent]:
    """Processes the streaming agent responses and returns TaskArtifactUpdateEvent and TaskStatusUpdateEvent."""
    is_task_complete = agent_response['is_task_complete']
    require_user_input = agent_response['require_user_input']
    parts: list[Part] = [Part(TextPart(text=agent_response['content']))]

    end_stream = False
    artifact = None
    message = None

    # responses from this agent can be working/completed/input-required
    if not is_task_complete and not require_user_input:
        task_state = TaskState.working
        message = Message(role=Role.agent, parts=parts, messageId=str(uuid4()))
    elif require_user_input:
        task_state = TaskState.input_required
        message = Message(role=Role.agent, parts=parts, messageId=str(uuid4()))
        end_stream = True
    else:
        task_state = TaskState.completed
        artifact = Artifact(parts=parts, artifactId=str(uuid4()))
        end_stream = True

    task_artifact_update_event = None

    if artifact:
        task_artifact_update_event = TaskArtifactUpdateEvent(
            taskId=task.id,
            contextId=task.contextId,
            artifact=artifact,
            append=False,
            lastChunk=True,
        )

    task_status_event = TaskStatusUpdateEvent(
        taskId=task.id,
        contextId=task.contextId,
        status=TaskStatus(
            state=task_state,
            message=message,
            timestamp=datetime.now().isoformat(),
        ),
        final=end_stream,
    )

    return task_artifact_update_event, task_status_event


def build_text_artifact(text: str, artifact_id: str) -> Artifact:
    """Helper to convert agent text to artifact."""
    text_part = TextPart(text=text)
    part = Part(text_part)
    return Artifact(parts=[part], artifactId=artifact_id)