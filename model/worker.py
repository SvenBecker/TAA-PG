def build_worker(agent):

    class Worker(agent):
        def __init__(self, model=None, **kwargs):
            # set model externally
            self.model = model
            super(Worker, self).__init__(**kwargs)

    return Worker


def clone_agent(agent=None, factor=None, environment=None, network=None, agent_config=None):
    worker = [agent]
    for i in range(factor - 1):
        worker = build_worker(type(agent))(
            states=environment.states,
            actions=environment.actions,
            network=network,
            model=agent.model,
            **agent_config
        )
        worker.append(worker)

    return worker