from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "mutation_operations"
            ADD COLUMN IF NOT EXISTS "verification_attempt_count" INT NOT NULL DEFAULT 0;
        DO $$
        DECLARE constraint_row RECORD;
        BEGIN
            FOR constraint_row IN
                SELECT c.conname
                FROM pg_constraint c
                JOIN pg_class t ON t.oid = c.conrelid
                WHERE t.relname = 'mutation_operations'
                  AND c.contype = 'c'
                  AND pg_get_constraintdef(c.oid) LIKE '%verification_status%'
            LOOP
                EXECUTE format(
                    'ALTER TABLE mutation_operations DROP CONSTRAINT %I',
                    constraint_row.conname
                );
            END LOOP;
        END $$;
        ALTER TABLE "mutation_operations"
            ADD CONSTRAINT "mutation_operations_verification_status_check"
            CHECK ("verification_status" IN (
                'not_required', 'pending', 'verifying', 'passed', 'failed',
                'inconclusive', 'needs_review'
            ));
    """


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        UPDATE "mutation_operations"
            SET "verification_status" = 'inconclusive'
            WHERE "verification_status" = 'needs_review';
        ALTER TABLE "mutation_operations"
            DROP CONSTRAINT IF EXISTS "mutation_operations_verification_status_check";
        ALTER TABLE "mutation_operations"
            ADD CONSTRAINT "mutation_operations_verification_status_check"
            CHECK ("verification_status" IN (
                'not_required', 'pending', 'verifying', 'passed', 'failed', 'inconclusive'
            ));
        ALTER TABLE "mutation_operations"
            DROP COLUMN IF EXISTS "verification_attempt_count";
    """
