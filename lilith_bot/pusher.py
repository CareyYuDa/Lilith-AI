# -*- coding: utf-8 -*-
import os, json, time, uuid, sqlite3, random
from typing import Optional
import httpx

OPENWEBUI_BASE_URL = os.getenv("OPENWEBUI_BASE_URL", "http://localhost:8080")
OPENWEBUI_API_KEY = os.getenv("OPENWEBUI_API_KEY", "")
LILITH_CHANNEL_NAME = os.getenv("LILITH_CHANNEL_NAME", "莉莉丝的房间")
LILITH_CHANNEL_DESC = "莉莉丝主动说话的地方"

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OWUI_DATA = os.path.join(_ROOT, "venv", "Lib", "site-packages", "open_webui", "data")
_USER_ID = None

class LilithPusher:
    def __init__(self, base_url=None, api_key=None):
        self.base_url = base_url or OPENWEBUI_BASE_URL
        self.api_key = api_key or OPENWEBUI_API_KEY
        self._channel_id = None
        self._headers = {"Authorization": "Bearer " + (self.api_key or ""), "Content-Type": "application/json"}
        self._user_id = None
        self._db_path = os.path.join(_OWUI_DATA, "webui.db")

    def _get_db(self):
        return sqlite3.connect(self._db_path)

    def _get_user_id(self):
        global _USER_ID
        if self._user_id: return self._user_id
        if _USER_ID: self._user_id = _USER_ID; return self._user_id
        try:
            conn = self._get_db()
            row = conn.execute("SELECT id FROM user LIMIT 1").fetchone()
            conn.close()
            if row: _USER_ID = self._user_id = row[0]
        except: pass
        return self._user_id

    def get_or_create_channel(self, name=None, description=None):
        name = name or LILITH_CHANNEL_NAME
        desc = description or LILITH_CHANNEL_DESC
        if self._channel_id: return self._channel_id
        try:
            channels = self._list_channels_rest()
            for ch in channels:
                if ch.get("name") == name:
                    self._channel_id = ch["id"]
                    return self._channel_id
            result = self._create_channel_rest(name, desc)
            self._channel_id = result.get("id")
        except:
            self._channel_id = self._create_channel_db(name, desc)
        return self._channel_id

    def _list_channels_rest(self):
        url = self.base_url + "/api/v1/channels/"
        with httpx.Client(timeout=30) as c:
            r = c.get(url, headers=self._headers)
            r.raise_for_status()
            return r.json() if r.text else []

    def _create_channel_rest(self, name, desc):
        url = self.base_url + "/api/v1/channels/create/"
        with httpx.Client(timeout=30) as c:
            r = c.post(url, json={"name": name, "description": desc}, headers=self._headers)
            r.raise_for_status()
            return r.json() if r.text else {}

    def _create_channel_db(self, name, description):
        uid = self._get_user_id()
        cid = str(uuid.uuid4())
        now_ns = int(time.time() * 1e9)
        conn = self._get_db()
        conn.execute("INSERT INTO channel (id, user_id, name, description, data, meta, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)", (cid, uid, name, description, "null", "null", now_ns, now_ns))
        conn.commit(); conn.close()
        return cid

    def send_message(self, content, channel_id=None):
        cid = channel_id or self._channel_id
        if not cid: cid = self.get_or_create_channel()
        uid = self._get_user_id()
        if not uid: raise RuntimeError("no user id")
        msg_id = str(uuid.uuid4())
        now_ns = int(time.time() * 1e9)
        conn = self._get_db()
        conn.execute("INSERT INTO message (id, user_id, channel_id, content, data, meta, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)", (msg_id, uid, cid, content, "null", "null", now_ns, now_ns))
        conn.execute("UPDATE channel SET updated_at=? WHERE id=?", (now_ns, cid))
        conn.commit(); conn.close()
        return {"id": msg_id, "content": content}

    send_message_stream = send_message

def get_pusher():
    return LilithPusher()

def push_to_channel(content):
    p = get_pusher()
    try: p.get_or_create_channel(); p.send_message(content); return True
    except Exception as e: print("[Pusher]", e); return False
