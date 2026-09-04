# Local infrastructure

The Compose project currently runs PostgreSQL for API persistence and LangGraph
checkpoints. PostgreSQL is published on loopback only and is not reachable from
other machines.

From the repository root:

```powershell
docker compose --env-file .env -f infra/docker/docker-compose.yml up -d postgres
docker compose -f infra/docker/docker-compose.yml ps
docker compose -f infra/docker/docker-compose.yml logs -f postgres
```

The first startup initializes the database from `postgres/schema.sql`. The API
uses the matching `DATABASE_URL` from the repository `.env` file. LangGraph
creates and migrates its own checkpoint tables when the checkpointer first
opens; those tables intentionally do not live in `schema.sql`.

Stop the service without deleting data:

```powershell
docker compose -f infra/docker/docker-compose.yml down
```

To discard all local PostgreSQL data and apply the initialization schema again:

```powershell
docker compose -f infra/docker/docker-compose.yml down --volumes
docker compose --env-file .env -f infra/docker/docker-compose.yml up -d postgres
```

`down --volumes` permanently deletes the local database volume. It is required
only when intentionally rebuilding the development database from scratch;
schema changes for persistent environments should use migrations.
