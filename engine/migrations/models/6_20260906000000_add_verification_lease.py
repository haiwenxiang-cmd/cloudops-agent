from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "mutation_operations"
            ADD COLUMN IF NOT EXISTS "verification_lease_owner" VARCHAR(160),
            ADD COLUMN IF NOT EXISTS "verification_lease_expires_at" TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS "approval_reason" TEXT;
        CREATE INDEX IF NOT EXISTS "idx_mutation_operations_verification_lease"
            ON "mutation_operations" (
                "verification_status", "verification_lease_expires_at"
            );
    """


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP INDEX IF EXISTS "idx_mutation_operations_verification_lease";
        ALTER TABLE "mutation_operations"
            DROP COLUMN IF EXISTS "approval_reason",
            DROP COLUMN IF EXISTS "verification_lease_expires_at",
            DROP COLUMN IF EXISTS "verification_lease_owner";
    """
