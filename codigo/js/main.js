document.addEventListener("DOMContentLoaded", () => {
  const chatbotBtn = document.getElementById("chatbot-btn");
  const chatbotBox = document.getElementById("chatbot-box");
  const chatBody = document.getElementById("chat-body");
  const chatInput = document.getElementById("chat-input");
  const sendBtn = document.getElementById("send-btn");

  if (!chatbotBtn || !chatbotBox || !chatBody || !chatInput || !sendBtn) {
    console.error("⚠ Algum elemento do chatbot não foi encontrado.");
    return;
  }

  chatbotBtn.addEventListener("click", () => {
    chatbotBox.classList.toggle("hidden");
    const isHidden = chatbotBox.classList.contains("hidden");
    chatbotBox.setAttribute("aria-hidden", isHidden);
  });

  let filtros = {};
  let caminho = [];
  let estadoAtual = "start";

  const fluxo = {
    start: {
      pergunta: "👋 Vamos achar um jogo pra você.\nPor onde quer começar?",
      opcoes: [
        { label: "Já sei o gênero", next: "genero" },
        { label: "Não sei, me guia", next: "humor" }
      ]
    },

    genero: {
      pergunta: "🎮 Escolha um gênero principal:",
      opcoes: [
        { label: "Ação", value: "acao", next: "plataforma" },
        { label: "Tiro / Shooter", value: "shooter", next: "plataforma" },
        { label: "RPG", value: "rpg", next: "plataforma" },
        { label: "Aventura", value: "aventura", next: "plataforma" },
        { label: "Estratégia", value: "estrategia", next: "plataforma" },
        { label: "Arcade / Plataforma", value: "plataforma", next: "plataforma" },
        { label: "Terror", value: "terror", next: "plataforma" },
        { label: "Indie", value: "indie", next: "plataforma" },
        { label: "Outro (digitar)", next: "genero_digitado" }
      ]
    },

    genero_digitado: {
      pergunta: "Digite o gênero ou estilos que você curte (ex: 'soulslike, corrida, esportes'):",
      input: "genero",
      next: "plataforma"
    },

    humor: {
      pergunta: "Que tipo de experiência você quer agora?",
      opcoes: [
        { label: "Relaxar / casual", value: "jogo leve e relaxante para familia", next: "plataforma" },
        { label: "História forte", value: "história forte e narrativa emocionante", next: "plataforma" },
        { label: "Competitivo", value: "competitivo e pvp online", next: "plataforma" },
        { label: "Co-op com amigos", value: "multiplayer co-op com amigos", next: "plataforma" },
        { label: "Assustador / terror", value: "terror assustador com zumbi", next: "plataforma" },
        { label: "Quebrar a cabeça", value: "puzzle com quebra-cabeca e enigma", next: "plataforma" }
      ]
    },

    plataforma: {
      pergunta: "📦 Em qual plataforma você quer jogar?",
      opcoes: [
        { label: "PC", value: "pc", next: "ano" },
        { label: "PlayStation", value: "playstation", next: "ano" },
        { label: "Xbox", value: "xbox", next: "ano" },
        { label: "Nintendo", value: "nintendo", next: "ano" },
        { label: "Mobile (Android / iOS)", value: "mobile", next: "ano" },
        { label: "Qualquer uma", value: "qualquer", next: "ano" }
      ]
    },

    ano: {
      pergunta: "📅 E em relação ao ano de lançamento?",
      opcoes: [
        { label: "Mais recentes (últimos anos)", value: "recentes", next: "nota" },
        { label: "Antes de 2015", value: "antes2015", next: "nota" },
        { label: "Antes de 2010", value: "antes2010", next: "nota" },
        { label: "Ano específico", next: "ano_digitado" },
        { label: "Tanto faz", value: "qualquer", next: "nota" }
      ]
    },

    ano_digitado: {
      pergunta: "Digite o ano desejado (ex: 2018):",
      input: "ano",
      next: "nota"
    },

    nota: {
      pergunta: "⭐ Quer definir uma nota mínima?",
      opcoes: [
        { label: "4.0 ou mais", value: "4.0", next: "faixa" },
        { label: "3.5 ou mais", value: "3.5", next: "faixa" },
        { label: "3.0 ou mais", value: "3.0", next: "faixa" },
        { label: "Não, tanto faz", value: "qualquer", next: "faixa" }
      ]
    },

    faixa: {
      pergunta: "🔞 Tem alguma restrição de faixa etária?",
      opcoes: [
        { label: "Qualquer", value: "qualquer", next: "tags" },
        { label: "Livre / família", value: "LIVRE", next: "tags" },
        { label: "+10", value: "+10", next: "tags" },
        { label: "+13", value: "+13", next: "tags" },
        { label: "+17", value: "+17", next: "tags" }
      ]
    },

    tags: {
      pergunta: "Quer adicionar alguma característica extra?",
      opcoes: [
        { label: "Não, pode seguir", value: "nenhuma", next: "confirmacao" },
        { label: "Mundo aberto", value: "mundo aberto exploracao aventura", next: "confirmacao" },
        { label: "Zumbis / Terror", value: "terror zumbi horror", next: "confirmacao" },
        { label: "Co-op / Multiplayer", value: "multiplayer co-op cooperativo online", next: "confirmacao" },
        { label: "Competitivo / PvP", value: "competitivo ranked pvp", next: "confirmacao" },
        { label: "Fantasia / Medieval", value: "fantasia medieval magia dragao", next: "confirmacao" },
        { label: "Puzzle / Quebra-cabeça", value: "puzzle quebra-cabeca enigma", next: "confirmacao" },
        { label: "Família / Casual", value: "familia leve relaxante kids", next: "confirmacao" },
        { label: "Indie / Alternativo", value: "indie pixel 2d metroidvania", next: "confirmacao" }
      ]
    },

    confirmacao: {
      pergunta: () => `
Confira o que você escolheu:<br><br>
🎮 Estilo / gênero: <b>${filtros.genero || filtros.humor || "não definido"}</b><br>
🖥 Plataforma: <b>${filtros.plataforma || "qualquer"}</b><br>
📅 Ano: <b>${filtros.ano || "qualquer"}</b><br>
⭐ Nota mínima: <b>${filtros.nota || "sem filtro"}</b><br>
🔞 Faixa etária: <b>${filtros.faixa || "qualquer"}</b><br>
🏷 Extras: <b>${filtros.tags || "nenhum"}</b><br><br>
Posso buscar jogos com base nisso?
      `,
      opcoes: [
        { label: "Sim, buscar jogos", next: "buscar" },
        { label: "Quero refazer os filtros", next: "start" }
      ]
    },

    buscar: {
      acao: "buscar"
    }
  };

  function addBotMessage(html) {
    const el = document.createElement("p");
    el.className = "bot";
    el.innerHTML = html;
    chatBody.appendChild(el);
    chatBody.scrollTop = chatBody.scrollHeight;
  }

  function addUserMessage(text) {
    const el = document.createElement("p");
    el.className = "user";
    el.textContent = text;
    chatBody.appendChild(el);
    chatBody.scrollTop = chatBody.scrollHeight;
  }

  function mostrarEstado(id) {
    estadoAtual = id;
    const node = fluxo[id];

    if (node && node.acao === "buscar") {
      buscarJogos();
      return;
    }

    if (!node) {
      console.error("Estado inválido:", id);
      return;
    }

    if (caminho[caminho.length - 1] !== id) {
      caminho.push(id);
    }

    chatBody.innerHTML = "";

    const pergunta =
      typeof node.pergunta === "function" ? node.pergunta() : node.pergunta;

    addBotMessage(pergunta);

    if (node.opcoes) {
      node.opcoes.forEach((op) => {
        const btn = document.createElement("button");
        btn.className = "option-btn";
        btn.textContent = "➡ " + op.label;

        btn.addEventListener("click", () => {
          if (op.value) salvarValor(id, op.value);
          mostrarEstado(op.next);
        });

        chatBody.appendChild(btn);
      });
    }

    if (id !== "start") {
      const back = document.createElement("button");
      back.className = "option-btn back";
      back.textContent = "⬅ Voltar";
      back.addEventListener("click", voltar);
      chatBody.appendChild(back);
    }

    chatBody.scrollTop = chatBody.scrollHeight;
  }

  function salvarValor(id, valor) {
    if (id === "genero" || id === "genero_digitado") filtros.genero = valor;
    if (id === "humor") filtros.humor = valor;
    if (id === "plataforma") filtros.plataforma = valor;
    if (id === "faixa") filtros.faixa = valor;
    if (id === "nota") filtros.nota = valor;
    if (id === "ano" || id === "ano_digitado") filtros.ano = valor;
    if (id === "tags") filtros.tags = valor;
  }

  function voltar() {
    caminho.pop();
    const anterior = caminho.pop() || "start";
    mostrarEstado(anterior);
  }

  sendBtn.addEventListener("click", processarTextoDigitado);
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") processarTextoDigitado();
  });

  function processarTextoDigitado() {
    const txt = chatInput.value.trim();
    if (!txt) return;

    const node = fluxo[estadoAtual];

    if (!node || !node.input) return;

    addUserMessage(txt);
    filtros[node.input] = txt;
    chatInput.value = "";

    mostrarEstado(node.next);
  }

  function resetFiltros() {
    filtros = {};
    caminho = [];
    estadoAtual = "start";

    chatBody.innerHTML = "";
    addBotMessage("🔄 Filtros reiniciados. Vamos começar de novo!");
    mostrarEstado("start");
  }

  function adicionarBotaoReset() {
    const resetBtn = document.createElement("button");
    resetBtn.className = "option-btn reset";
    resetBtn.textContent = "🔄 Reiniciar recomendação";
    resetBtn.addEventListener("click", resetFiltros);

    chatBody.appendChild(resetBtn);
    chatBody.scrollTop = chatBody.scrollHeight;
  }

  async function buscarJogos() {
    const blocoTags = `
[GENERO=${filtros.genero || "qualquer"}]
[HUMOR=${filtros.humor || "qualquer"}]
[PLATAFORMA=${filtros.plataforma || "qualquer"}]
[FAIXA=${filtros.faixa || "qualquer"}]
[NOTA=${filtros.nota || "qualquer"}]
[ANO=${filtros.ano || "qualquer"}]
[TAGS=${filtros.tags || "nenhuma"}]
`.trim();

    const fraseNatural = `
Quero jogos do gênero ${filtros.genero || filtros.humor || "qualquer"} 
para jogar em ${filtros.plataforma || "qualquer plataforma"}, 
com nota mínima ${filtros.nota || "sem filtro"},
faixa etária ${filtros.faixa || "qualquer"},
ano ${filtros.ano || "qualquer"},
com essas características extras: ${filtros.tags || "nenhuma"}.
    `.trim();

    const mensagem = blocoTags + "\n\n" + fraseNatural;

    console.log("📨 Enviando para /chat:\n", mensagem);

    addBotMessage("🔍 Buscando jogos compatíveis com as suas escolhas...");

    try {
      const resp = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mensagem })
      });

      if (!resp.ok) {
        chatBody.innerHTML = "";
        addBotMessage("❌ Erro ao buscar jogos no servidor.");
        adicionarBotaoReset();
        return;
      }

      const data = await resp.json();
      let jogos = data.resposta || [];
      let intro = "";

      if (jogos[0]?.intro) {
        intro = jogos[0].intro;
        jogos = jogos.slice(1);
      }

      if (!Array.isArray(jogos) || jogos.length === 0) {
        chatBody.innerHTML = "";
        addBotMessage(
          (intro || "⚠ Não encontramos jogos com esses filtros.") +
          "<br><br>Tente mudar gênero, plataforma ou ano para ampliar a busca."
        );
        adicionarBotaoReset();
        return;
      }

      let html = `
        <div style="margin-bottom: 12px; white-space: pre-line;">
          ${intro || "🎮 Aqui estão algumas recomendações para você:"}
        </div>
        <ul style="list-style:none; padding:0; margin:0;">
      `;

      jogos.forEach((j) => {
        html += `
          <li style="margin-bottom:14px; border-bottom:1px solid #1b2838; padding-bottom:10px;">
            <div style="font-weight:bold; font-size:14px;">
              ${j.nome} <span style="color:#66c0f4;">(${j.genero || "Gênero não informado"})</span>
            </div>

            <small>
              <b>Ano:</b> ${j.ano || "—"} • 
              <b>Plataforma:</b> ${j.plataforma || "—"}
            </small><br>
            <small><b>Publisher:</b> ${j.publisher || "—"}</small><br>
            <small><b>Nota:</b> ⭐ ${j.nota?.toFixed ? j.nota.toFixed(1) : j.nota}</small><br>
            <small><b>Faixa etária:</b> ${j.faixa_etaria || "N/A"}</small><br>
            <small style="color:#bbb; display:block; margin-top:4px;">
              <b>Descrição:</b> ${j.descricao || "Sem descrição disponível."}
            </small>
          </li>
        `;
      });

      html += `</ul>`;

      chatBody.innerHTML = "";
      addBotMessage(html);
      adicionarBotaoReset();
    } catch (err) {
      console.error("Erro ao chamar /chat:", err);
      chatBody.innerHTML = "";
      addBotMessage("❌ Ocorreu um erro ao conectar ao servidor.");
      adicionarBotaoReset();
    }
  }

  mostrarEstado("start");
});
