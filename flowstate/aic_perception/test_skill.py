#!/usr/bin/env python3
"""Stateless Flowstate lifecycle probe.

This deliberately does not import rclpy, OpenCV, NumPy, AIC model code,
camera topics, TF, controller APIs, or persistent state.  It exists solely to
separate the Flowstate skill-image/workload lifecycle from board perception.
"""

from __future__ import annotations

from concurrent import futures
import signal

from absl import app, flags, logging
import grpc

from intrinsic.skills.python import skill_interface


FLAGS = flags.FLAGS
flags.DEFINE_integer("port", 8003, "Port to listen on.", allow_override=True)
flags.DEFINE_string(
    "skill_service_config_filename",
    "",
    "Path to the generated skill config.",
    allow_override=True,
)


class TestSkill(skill_interface.Skill):
    """A no-state skill whose successful call proves the service is alive."""

    def configure_runtime(self, service_config) -> None:
        from intrinsic.skills.internal import runtime_data
        from aic_perception import test_skill_pb2 as pb2

        self._skill_alias = service_config.skill_description.skill_name
        self._runtime_data = runtime_data.get_runtime_data_from(
            service_config,
            pb2.TestSkillParams.DESCRIPTOR,
        )

    def _check_alias(self, name: str) -> None:
        if name != self._skill_alias:
            from intrinsic.skills.internal import skill_repository

            raise skill_repository.InvalidSkillAliasError(
                f"unknown skill alias: {name}"
            )

    def get_skill(self, name):
        self._check_alias(name)
        return self

    def get_skill_execute(self, name):
        self._check_alias(name)
        return self

    def get_skill_project(self, name):
        self._check_alias(name)
        return self

    def get_skill_runtime_data(self, name):
        self._check_alias(name)
        return self._runtime_data

    def get_skill_aliases(self):
        return [self._skill_alias]

    def execute(self, request, context):
        from aic_perception import test_skill_pb2 as pb2

        context.canceller.ready()
        logging.info("TestSkill execute: lifecycle probe succeeded")
        return pb2.TestSkillResult(
            success=True,
            message="lifecycle probe ok",
        )


def start_runner(argv):
    """Start only the three standard Flowstate gRPC services."""
    del argv
    logging.info("TestSkill service starting")
    from intrinsic.skills.internal import skill_service_impl
    from intrinsic.skills.proto import skill_service_config_pb2
    from intrinsic.skills.proto import skill_service_pb2_grpc

    if not FLAGS.skill_service_config_filename:
        raise ValueError("--skill_service_config_filename is required")
    service_config = skill_service_config_pb2.SkillServiceConfig()
    with open(FLAGS.skill_service_config_filename, "rb") as config_file:
        service_config.ParseFromString(config_file.read())

    skill_instance = TestSkill()
    skill_instance.configure_runtime(service_config)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    skill_service_pb2_grpc.add_ProjectorServicer_to_server(
        skill_service_impl.SkillProjectorServicer(skill_instance, None, None, None),
        server,
    )
    skill_service_pb2_grpc.add_ExecutorServicer_to_server(
        skill_service_impl.SkillExecutorServicer(skill_instance, None, None, None),
        server,
    )
    skill_service_pb2_grpc.add_SkillInformationServicer_to_server(
        skill_service_impl.SkillInformationServicer(service_config.skill_description),
        server,
    )
    server.add_insecure_port(f"[::]:{FLAGS.port}")
    server.start()
    logging.info("gRPC server listening on port %s", FLAGS.port)

    def stop_service(signum, _frame):
        logging.info("TestSkill stopping on signal %s", signum)
        server.stop(grace=1.0)

    signal.signal(signal.SIGINT, stop_service)
    signal.signal(signal.SIGTERM, stop_service)
    try:
        server.wait_for_termination()
    finally:
        server.stop(grace=0)
        logging.info("TestSkill stopped")


if __name__ == "__main__":
    app.run(start_runner, flags_parser=lambda argv: FLAGS(argv, known_only=True))
