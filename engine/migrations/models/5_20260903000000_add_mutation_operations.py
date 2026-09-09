from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "mutation_operations" (
            "id" UUID NOT NULL PRIMARY KEY,
            "idempotency_key" VARCHAR(160) NOT NULL UNIQUE,
            "workspace_id" UUID,
            "conversation_id" VARCHAR(160),
            "run_id" VARCHAR(160),
            "call_id" VARCHAR(160),
            "user_id" UUID,
            "user_role" VARCHAR(32),
            "tool_name" VARCHAR(160) NOT NULL,
            "tool_category" VARCHAR(40) NOT NULL,
            "target_identity" JSONB NOT NULL DEFAULT '{}',
            "redacted_args" JSONB NOT NULL DEFAULT '{}',
            "args_hmac" VARCHAR(64) NOT NULL,
            "desired_state_hmac" VARCHAR(64),
            "policy_decision" VARCHAR(20) NOT NULL
                CHECK ("policy_decision" IN ('allow', 'deny')),
            "policy_reason" TEXT NOT NULL DEFAULT '',
            "approval_status" VARCHAR(20) NOT NULL DEFAULT 'pending'
                CHECK ("approval_status" IN ('not_required', 'pending', 'approved', 'denied')),
            "approved_by_user_id" UUID,
            "approved_at" TIMESTAMPTZ,
            "execution_status" VARCHAR(32) NOT NULL DEFAULT 'prepared'
                CHECK ("execution_status" IN (
                    'prepared', 'executing', 'reported_success', 'failed', 'unknown', 'cancelled'
                )),
            "verification_status" VARCHAR(32) NOT NULL DEFAULT 'pending'
                CHECK ("verification_status" IN (
                    'not_required', 'pending', 'verifying', 'passed', 'failed', 'inconclusive'
                )),
            "attempt_count" INT NOT NULL DEFAULT 0,
            "external_reference" JSONB NOT NULL DEFAULT '{}',
            "execution_result" JSONB NOT NULL DEFAULT '{}',
            "execution_error" JSONB NOT NULL DEFAULT '{}',
            "verification_plan" JSONB NOT NULL DEFAULT '{}',
            "verification_result" JSONB NOT NULL DEFAULT '{}',
            "lease_owner" VARCHAR(160),
            "lease_expires_at" TIMESTAMPTZ,
            "started_at" TIMESTAMPTZ,
            "finished_at" TIMESTAMPTZ,
            "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            "version" INT NOT NULL DEFAULT 1
        );
        CREATE INDEX IF NOT EXISTS "idx_mutation_operations_workspace_id"
            ON "mutation_operations" ("workspace_id");
        CREATE INDEX IF NOT EXISTS "idx_mutation_operations_conversation_id"
            ON "mutation_operations" ("conversation_id");
        CREATE INDEX IF NOT EXISTS "idx_mutation_operations_run_id"
            ON "mutation_operations" ("run_id");
        CREATE INDEX IF NOT EXISTS "idx_mutation_operations_call_id"
            ON "mutation_operations" ("call_id");
        CREATE INDEX IF NOT EXISTS "idx_mutation_operations_user_id"
            ON "mutation_operations" ("user_id");
        CREATE INDEX IF NOT EXISTS "idx_mutation_operations_tool_name"
            ON "mutation_operations" ("tool_name");
        CREATE INDEX IF NOT EXISTS "idx_mutation_operations_execution_lease"
            ON "mutation_operations" ("execution_status", "lease_expires_at");
        CREATE INDEX IF NOT EXISTS "idx_mutation_operations_verification_updated"
            ON "mutation_operations" ("verification_status", "updated_at");
        CREATE INDEX IF NOT EXISTS "idx_mutation_operations_conversation_created"
            ON "mutation_operations" ("conversation_id", "created_at");
    """


async def downgrade(db: BaseDBAsyncClient) -> str:
    return 'DROP TABLE IF EXISTS "mutation_operations";'
