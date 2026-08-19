from __future__ import annotations

from typing import Final

import gradio as gr

from novel_agent.contracts import RunCommandType
from novel_agent.service import NovelAgentService
from novel_agent.ui_support import (
    RunUiController,
    reset_rebase_controls,
    reset_run_controls,
    run_control_updates,
)


_run_control_updates = run_control_updates


_CONSOLE_CSS: Final = """
@media (max-width: 640px) {
  #governed-console {
    flex-direction: column !important;
  }

  #governed-console > .gr-column {
    min-width: 0 !important;
    width: 100% !important;
  }
}
"""


def create_novel_app(service: NovelAgentService | None = None) -> gr.Blocks:
    managed_service = service or NovelAgentService.from_config()
    controller = RunUiController(managed_service)
    project_choices = [project_id.root for project_id in managed_service.project_ids()]
    default_project = project_choices[0]

    with gr.Blocks(title="受治理小说生成 Agent") as app:
        gr.Markdown("# 受治理小说生成 Agent")
        with gr.Row(elem_id="governed-console"):
            with gr.Column(scale=1):
                gr.Markdown("## 运行控制")
                project_dropdown = gr.Dropdown(
                    choices=project_choices,
                    value=default_project,
                    label="项目",
                    interactive=True,
                )
                run_id_textbox = gr.Textbox(label="运行标识", placeholder="例如：run-001")
                requirements_textbox = gr.Textbox(label="创作需求", lines=8)
                provider_dropdown = gr.Dropdown(
                    choices=["offline_fake", "legacy_real"],
                    value="offline_fake",
                    label="提供方",
                    interactive=True,
                )
                start_button = gr.Button("启动运行", variant="primary")
                with gr.Row():
                    pause_button = gr.Button("暂停", interactive=False)
                    resume_button = gr.Button("继续", interactive=False)
                    cancel_button = gr.Button("取消", variant="stop", interactive=False)
                with gr.Row():
                    repair_button = gr.Button("修复并重新排队", interactive=False)
                    refresh_button = gr.Button("刷新状态")
                with gr.Row():
                    workflow_button = gr.Button("刷新工作流视图", variant="secondary")
                command_result = gr.Textbox(label="命令结果", interactive=False, lines=3)
                snapshot_textbox = gr.Textbox(label="运行快照", interactive=False, lines=12)
                event_textbox = gr.Textbox(label="运行事件", interactive=False, lines=12)

                gr.Markdown("## Agent 工作流程")
                workflow_status = gr.Textbox(label="当前阶段", interactive=False, lines=2)
                workflow_graph = gr.HTML(label="任务图进度")
                workflow_chain = gr.Textbox(label="思维链（可审计执行轨迹）", interactive=False, lines=18)

            with gr.Column(scale=1):
                gr.Markdown("## 受治理内容")
                artifact_id_textbox = gr.Textbox(label="Artifact 标识", placeholder="例如：chapter-001")
                artifact_revision = gr.Number(label="版本", value=1, precision=0, minimum=1)
                artifact_button = gr.Button("查看版本")
                artifact_metadata = gr.Textbox(label="Artifact 元数据", interactive=False, lines=4)
                artifact_text = gr.Textbox(label="Artifact 内容", interactive=False, lines=12)
                audit_button = gr.Button("刷新审计与质量")
                audit_text = gr.Textbox(label="审计与质量", interactive=False, lines=12)

        with gr.Accordion("分支重建确认", open=False):
            rebase_confirmation = gr.Textbox(label="确认载荷", lines=10)
            rebase_button = gr.Button("确认分支重建", variant="stop")

        rebase_confirmation.interactive = False
        rebase_button.interactive = False

        run_outputs = [
            command_result,
            snapshot_textbox,
            event_textbox,
            pause_button,
            resume_button,
            cancel_button,
            repair_button,
        ]
        project_dropdown.change(fn=reset_run_controls, inputs=None, outputs=run_outputs[3:]).then(
            fn=reset_rebase_controls,
            inputs=None,
            outputs=[rebase_confirmation, rebase_button],
        )
        run_id_textbox.change(fn=reset_run_controls, inputs=None, outputs=run_outputs[3:]).then(
            fn=reset_rebase_controls,
            inputs=None,
            outputs=[rebase_confirmation, rebase_button],
        )
        start_button.click(
            fn=controller.start_run,
            inputs=[project_dropdown, run_id_textbox, requirements_textbox, provider_dropdown],
            outputs=run_outputs,
        ).then(
            fn=controller.refresh_rebase_controls,
            inputs=[project_dropdown, run_id_textbox],
            outputs=[rebase_confirmation, rebase_button],
        )
        pause_button.click(
            fn=lambda project_name, run_id: controller.control_run(project_name, run_id, RunCommandType.PAUSE),
            inputs=[project_dropdown, run_id_textbox],
            outputs=run_outputs,
        ).then(
            fn=controller.refresh_rebase_controls,
            inputs=[project_dropdown, run_id_textbox],
            outputs=[rebase_confirmation, rebase_button],
        )
        resume_button.click(
            fn=lambda project_name, run_id: controller.control_run(project_name, run_id, RunCommandType.RESUME),
            inputs=[project_dropdown, run_id_textbox],
            outputs=run_outputs,
        ).then(
            fn=controller.refresh_rebase_controls,
            inputs=[project_dropdown, run_id_textbox],
            outputs=[rebase_confirmation, rebase_button],
        )
        cancel_button.click(
            fn=lambda project_name, run_id: controller.control_run(project_name, run_id, RunCommandType.CANCEL),
            inputs=[project_dropdown, run_id_textbox],
            outputs=run_outputs,
        ).then(
            fn=controller.refresh_rebase_controls,
            inputs=[project_dropdown, run_id_textbox],
            outputs=[rebase_confirmation, rebase_button],
        )
        repair_button.click(
            fn=controller.repair_run,
            inputs=[project_dropdown, run_id_textbox],
            outputs=run_outputs,
        ).then(
            fn=controller.refresh_rebase_controls,
            inputs=[project_dropdown, run_id_textbox],
            outputs=[rebase_confirmation, rebase_button],
        )
        refresh_button.click(
            fn=controller.refresh_run,
            inputs=[project_dropdown, run_id_textbox],
            outputs=run_outputs,
        ).then(
            fn=controller.refresh_rebase_controls,
            inputs=[project_dropdown, run_id_textbox],
            outputs=[rebase_confirmation, rebase_button],
        )
        workflow_button.click(
            fn=controller.refresh_workflow,
            inputs=[project_dropdown, run_id_textbox],
            outputs=[workflow_status, workflow_graph, workflow_chain],
        )
        artifact_button.click(
            fn=controller.view_artifact,
            inputs=[project_dropdown, run_id_textbox, artifact_id_textbox, artifact_revision],
            outputs=[artifact_metadata, artifact_text],
        )
        audit_button.click(
            fn=controller.refresh_audit,
            inputs=[project_dropdown, run_id_textbox],
            outputs=[command_result, audit_text],
        )
        rebase_button.click(
            fn=controller.confirm_rebase,
            inputs=[project_dropdown, run_id_textbox, rebase_confirmation],
            outputs=run_outputs,
        ).then(
            fn=controller.refresh_rebase_controls,
            inputs=[project_dropdown, run_id_textbox],
            outputs=[rebase_confirmation, rebase_button],
        )
    return app


def launch_novel_app() -> None:
    create_novel_app().launch(css=_CONSOLE_CSS)


if __name__ == "__main__":
    launch_novel_app()
