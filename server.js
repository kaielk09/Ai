require("dotenv").config();
const express = require("express");
const path = require("path");

const app = express();
app.disable("x-powered-by");
app.use(express.json({ limit: "1mb" }));
app.use(express.static(path.join(__dirname, "public")));

const PORT = Number(process.env.PORT || 3000);
const API_BASE_URL = (process.env.API_BASE_URL || "").replace(/\/+$/, "");
const API_KEY = process.env.API_KEY || "";
const MODEL_NAME = process.env.MODEL_NAME || "";
const SEARCH_API_KEY = process.env.SEARCH_API_KEY || "";

app.get("/api/status", (_req, res) => {
  res.json({
    name: "Purple AI",
    version: "1.0.0",
    configured: Boolean(API_BASE_URL && API_KEY && MODEL_NAME),
    webSearchConfigured: Boolean(SEARCH_API_KEY)
  });
});

async function searchWeb(query) {
  if (!SEARCH_API_KEY) {
    throw new Error("Wyszukiwanie wymaga klucza SEARCH_API_KEY (Tavily).");
  }
  const response = await fetch("https://api.tavily.com/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      api_key: SEARCH_API_KEY,
      query,
      search_depth: "basic",
      max_results: 5,
      include_answer: false
    })
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.message || "Wyszukiwanie w sieci nie powiodło się.");
  return (data.results || []).map((item) => ({
    title: item.title || "Wynik",
    url: item.url || "",
    content: item.content || ""
  }));
}

app.post("/api/search", async (req, res) => {
  try {
    const query = String(req.body?.query || "").trim();
    if (!query || query.length > 500) return res.status(400).json({ error: "Wpisz zapytanie (maks. 500 znaków)." });
    const results = await searchWeb(query);
    res.json({ results });
  } catch (error) {
    res.status(502).json({ error: error.message || "Błąd wyszukiwania." });
  }
});

app.post("/api/chat", async (req, res) => {
  try {
    if (!API_BASE_URL || !API_KEY || !MODEL_NAME) {
      return res.status(503).json({ error: "AI nie jest skonfigurowane. Ustaw API_BASE_URL, API_KEY i MODEL_NAME w zmiennych środowiskowych hostingu." });
    }

    const messages = req.body?.messages;
    if (!Array.isArray(messages) || messages.length === 0 || messages.length > 40) {
      return res.status(400).json({ error: "Nieprawidłowa historia rozmowy." });
    }

    const safeMessages = messages.map((m) => ({
      role: ["user", "assistant"].includes(m?.role) ? m.role : "user",
      content: String(m?.content || "").slice(0, 12000)
    }));

    const response = await fetch(API_BASE_URL + "/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + API_KEY
      },
      body: JSON.stringify({
        model: MODEL_NAME,
        messages: [
          { role: "system", content: "Jesteś Purple AI v1.0, pomocnym asystentem odpowiadającym po polsku, chyba że użytkownik poprosi o inny język. Pomagaj w nauce i programowaniu. Nie twierdź, że coś sprawdziłeś w internecie, jeśli nie otrzymałeś wyników wyszukiwania." },
          ...safeMessages
        ],
        temperature: 0.7
      })
    });

    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const message = data?.error?.message || data?.message || "Dostawca API zwrócił błąd (" + response.status + ").";
      return res.status(502).json({ error: message });
    }

    const answer = data?.choices?.[0]?.message?.content;
    if (!answer) return res.status(502).json({ error: "API nie zwróciło odpowiedzi. Sprawdź zgodność API i nazwę modelu." });
    res.json({ answer });
  } catch (error) {
    res.status(502).json({ error: error.message || "Nie udało się połączyć z API AI." });
  }
});

app.get("*", (_req, res) => res.sendFile(path.join(__dirname, "public", "index.html")));

app.listen(PORT, "0.0.0.0", () => {
  console.log("Purple AI v1.0 działa na porcie " + PORT);
});