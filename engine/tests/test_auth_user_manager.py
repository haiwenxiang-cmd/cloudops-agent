from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from fastapi_users.exceptions import UserNotExists

from src.api.models.user import User
from src.api.services.auth import UserManager


class UserManagerEmailLookupTests(IsolatedAsyncioTestCase):
    async def test_missing_user_raises_fastapi_users_domain_exception(self) -> None:
        manager = UserManager(AsyncMock())
        lookup = AsyncMock(return_value=None)

        with patch.object(User, "get_or_none", new=lookup):
            with self.assertRaises(UserNotExists):
                await manager.get_by_email("missing@example.com")

        lookup.assert_awaited_once_with(email="missing@example.com")

    async def test_existing_user_is_returned(self) -> None:
        manager = UserManager(AsyncMock())
        existing_user = AsyncMock(spec=User)
        lookup = AsyncMock(return_value=existing_user)

        with patch.object(User, "get_or_none", new=lookup):
            result = await manager.get_by_email("existing@example.com")

        self.assertIs(result, existing_user)
        lookup.assert_awaited_once_with(email="existing@example.com")
