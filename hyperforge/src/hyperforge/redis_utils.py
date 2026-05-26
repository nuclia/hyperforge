from typing import Dict, Optional

from redis.asyncio.cluster import RedisCluster


class ManualStreamKeysRedisCluster(RedisCluster):
    """When connecting to a cluster, we need to know the key of each command to route it to the correct node. Stream functions (XREADGROUP, XREAD, etc) have a complicated syntax and the client does not extract the key, rather it sends the query to the server for parsing. If the server is down, this fails and the library does not have retries/reconnects for this code path. This can cause the client to get stuck and never reconnect. Since we know the keys, we can do this calculation ourselves to workaround this problem."""

    def get_node_id_from_key(self, key: str):
        keyslot = self.keyslot(key)
        return self.nodes_manager.get_node_from_slot(keyslot)

    @classmethod
    def from_url(cls, url, **kwargs):
        return super().from_url(
            url=url,
            **kwargs,
        )

    async def xreadgroup(
        self,
        groupname: str,
        consumername: str,
        streams: Dict[str, str],
        block: int,
        count: int,
        noack: bool,
    ):
        # Get the node from the first stream key
        if len(streams) != 1:
            raise ValueError("Only one stream key is supported")
        node = self.get_node_from_key(list(streams.keys())[0])
        cmd_args = [
            "XREADGROUP",
            "GROUP",
            groupname,
            consumername,
        ]
        if noack:
            cmd_args.append("NOACK")
        cmd_args.extend(
            [
                "BLOCK",
                str(block),
                "COUNT",
                str(count),
                "STREAMS",
                *streams.keys(),
                *streams.values(),
            ]
        )

        return await self.execute_command(*cmd_args, target_nodes=[node])

    async def xread(
        self,
        streams: Dict[str, str],
        block: int,
        count: Optional[int] = None,
    ):
        # Get the node from the first stream key
        # Get the node from the first stream key
        if len(streams) != 1:
            raise ValueError("Only one stream key is supported")
        node = self.get_node_from_key(list(streams.keys())[0])

        cmd_args = [
            "XREAD",
            "BLOCK",
            str(block),
        ]
        if count is not None:
            cmd_args.extend(["COUNT", str(count)])

        cmd_args.extend(
            [
                "STREAMS",
                *streams.keys(),
                *streams.values(),
            ]
        )
        return await self.execute_command(*cmd_args, target_nodes=[node])
