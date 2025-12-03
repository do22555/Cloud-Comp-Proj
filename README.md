# Cloud-Comp-Proj
SCIFM0004 Software Engineering and HPC cloud computing project repository.

## To Run:
1) Before running anything the containers must be built:
`docker compose build`
This builds three images: worker, producer and aggregator.

2) Start RabbitMQ before producer and worker as they depend on it. Use `-d` flag to run in detached mode. This launches the message broker.
`docker compose up -d rabbitmq`

3) Start the worker cluster in detached mode. Any reasonable number works here but I would choose a number like 4. 
`docker compose up -d --scale worker=<NUM_WORKERS>`
e.g.
`docker compose up -d --scale worker=4`
All workers will connect to RabbitMQ immediately and wait for tasks.

5) Run the producer: This discovers the dataset through atlasopenmagic, builds the file list, and queues one message per ROOT file:
`docker compose run --rm producer`
The data processing takes approximately 5 minutes. Users can watch the processing live with
`docker compose logs -f worker`
Workers continue until the tasks queue is empty.

6) Once all workers complete run the aggregator:
`docker compose run --rm aggregator`   This loads every file into [data/output] and produces [data/output/final_plot.png] which can be copied back to the local machine using `docker cp`.

## Summary:
```
docker compose build
docker compose up -d rabbitmq
docker compose up -d --scale worker=<NUM_WORKERS>
docker compose run --rm producer
docker compose logs -f worker
docker compose run --rm aggregator
docker cp <CONTAINER_ID>:/data/output ./output_local
```
