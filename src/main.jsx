import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { Analytics } from "@vercel/analytics/react";
import {
  Bot,
  Send,
  Sparkles,
  Upload
} from "lucide-react";
import "./styles.css";

const actions = [
  { id: "chat", label: "Talk with the artist" },
  { id: "paint", label: "Paint something" },
  { id: "redesign", label: "Redesign a picture" }
];

const toolCopy = {
  chat: {
    title: "Ask questions about their life and work",
    description:
      "Remember: this chatbot is not the artist; it only retrieves public information about them."
  },
  paint: {
    title: "Generate an artwork from a prompt",
    description: "Describe the image you want to create and the app will generate it in the selected style."
  },
  redesign: {
    title: "Transform your own image",
    description: "Upload a picture and reinterpret it using the selected artist's visual style."
  }
};

const API_FAILURE_MESSAGE =
  "The token quota for today has been reached or the external AI service is temporarily unavailable. Please try again later.";

function getArtistDisplayName(artist) {
  return artist.displayName || artist.name;
}

async function apiJson(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || "Request failed.");
  }
  return payload;
}

function getApiErrorMessage(error) {
  const message = error?.message || "";
  if (message.includes("Question is required") || message.includes("Upload a JPG or PNG image")) {
    return message;
  }

  return API_FAILURE_MESSAGE;
}

function App() {
  const [artists, setArtists] = useState([]);
  const [heroImages, setHeroImages] = useState([]);
  const [selectedArtist, setSelectedArtist] = useState(null);
  const [activeAction, setActiveAction] = useState("");
  const [showHero, setShowHero] = useState(true);
  const [pendingSection, setPendingSection] = useState("");
  const [loadError, setLoadError] = useState("");

  useEffect(() => {
    apiJson("/api/artists")
      .then(setArtists)
      .catch((error) => setLoadError(error.message));
    apiJson("/api/hero-images")
      .then(setHeroImages)
      .catch(() => setHeroImages([]));
  }, []);

  const goToLandingSection = (sectionId) => {
    setSelectedArtist(null);
    setPendingSection(sectionId);
    setShowHero(true);
  };

  const enterStudio = () => {
    window.scrollTo(0, 0);
    setShowHero(false);
  };

  const selectArtist = (artist) => {
    window.scrollTo(0, 0);
    setActiveAction("");
    setSelectedArtist(artist);
  };

  if (loadError) {
    return <main className="status-page">Could not load artists: {loadError}</main>;
  }

  if (showHero) {
    return (
      <HeroPage
        images={heroImages}
        initialSection={pendingSection}
        onSectionHandled={() => setPendingSection("")}
        onEnter={enterStudio}
      />
    );
  }

  if (!selectedArtist) {
    return (
        <ArtistGallery
          artists={artists}
        onSelect={selectArtist}
        onHome={() => setShowHero(true)}
        onAbout={() => goToLandingSection("about")}
        onStudio={enterStudio}
        onContact={() => goToLandingSection("contact")}
      />
    );
  }

  return (
    <Studio
      artist={selectedArtist}
      activeAction={activeAction}
      onActionChange={setActiveAction}
      onBack={() => setSelectedArtist(null)}
      onHome={() => {
        setSelectedArtist(null);
        setShowHero(true);
      }}
      onAbout={() => goToLandingSection("about")}
      onStudio={() => {
        setSelectedArtist(null);
        enterStudio();
      }}
      onContact={() => goToLandingSection("contact")}
    />
  );
}

function HeroPage({ images, initialSection, onSectionHandled, onEnter }) {
  const [activeImage, setActiveImage] = useState(0);
  const aboutRef = useRef(null);
  const contactRef = useRef(null);

  useEffect(() => {
    if (images.length <= 1) return undefined;

    const timer = window.setInterval(() => {
      setActiveImage((current) => (current + 1) % images.length);
    }, 4500);

    return () => window.clearInterval(timer);
  }, [images.length]);

  useEffect(() => {
    setActiveImage(0);
  }, [images.length]);

  useEffect(() => {
    if (!initialSection) return;

    window.requestAnimationFrame(() => {
      document.getElementById(initialSection)?.scrollIntoView({ behavior: "smooth", block: "start" });
      onSectionHandled();
    });
  }, [initialSection, onSectionHandled]);

  const scrollToSection = (sectionRef) => {
    sectionRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <main className="landing-page">
      <Header
        onHome={() => window.scrollTo({ top: 0, behavior: "smooth" })}
        onAbout={() => scrollToSection(aboutRef)}
        onStudio={onEnter}
        onContact={() => scrollToSection(contactRef)}
      />
      <section className="hero-page">
        <div className="hero-background" aria-hidden="true">
          {images.map((image, index) => (
            <img
              key={image.name}
              className={index === activeImage ? "active" : ""}
              src={image.imageUrl}
              alt=""
            />
          ))}
        </div>
        <section className="hero-content">
          <h1>ARTROOM AI</h1>
          <p>Explore the lives and styles of iconic artists through conversation and AI-generated images.</p>

          <button className="primary-button" onClick={onEnter} type="button">
            Enter Studio
          </button>
        </section>
        {images.length > 1 && (
          <div className="hero-dots" aria-label="Hero images">
            {images.map((image, index) => (
              <button
                key={image.name}
                className={index === activeImage ? "active" : ""}
                onClick={() => setActiveImage(index)}
                type="button"
                aria-label={`Show hero image ${index + 1}`}
              />
            ))}
          </div>
        )}
      </section>

      <section className="landing-section" id="about" ref={aboutRef}>
        <p className="eyebrow">ABOUT</p>
        <h2>Explore artists through AI-powered conversation and generation.</h2>
         <p style={{ textAlign: "justify", lineHeight: 1.5 }}>
          ArtRoom AI is a mock digital studio where visitors can ask questions, explore
          artist context, and generate images inspired by the work of famous painters.
          Developed in 2025 for a Generative AI course at the University of Chicago, the
          project uses open and freely available information and images as its source material.
          <br /><br />

          It uses GPT-4o with LangChain, Wikipedia-based RAG stored in FAISS for grounded chatbot 
          answers, and Stable Diffusion models, including a fine-tuned Pancho Fierro model 
          trained on 30 paintings, for image generation.
        </p>

      </section>

      <section className="landing-section contact-section" id="contact" ref={contactRef}>
        <p className="eyebrow">CONTACT</p>
        <h2>Get in touch.</h2>
        <p>
          If you would like to learn more about this project, review the materials,
          or get in touch, write to <a href="mailto:email">jpajuelo@uchicago.edu</a>
          
        </p>
      </section>
    </main>
  );
}

function Header({ onHome, onAbout, onStudio, onContact }) {
  const scrollToPageSection = (sectionId) => {
    document.getElementById(sectionId)?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <header className="site-header" aria-label="Site header">
      <button className="site-link home-link" onClick={onHome} type="button">
        <span>ArtRoom AI</span>
      </button>
      <nav className="site-nav" aria-label="Main navigation">
        <button className="site-link" onClick={onAbout || (() => scrollToPageSection("about"))} type="button">
          About
        </button>
        <button className="site-link" onClick={onStudio} type="button">
          Studio
        </button>
        <button className="site-link" onClick={onContact || (() => scrollToPageSection("contact"))} type="button">
          Contact
        </button>
      </nav>
    </header>
  );
}

function ArtistGallery({ artists, onSelect, onHome, onAbout, onStudio, onContact }) {
  return (
    <main className="app-shell gallery-shell">
      <Header onHome={onHome} onAbout={onAbout} onStudio={onStudio} onContact={onContact} />
      <header className="page-header">
        <h1>Welcome to the Studio</h1>
        <h2>
          Choose an artist, explore their life and work through conversation, and generate
          images inspired by their unique style.
        </h2>
      </header>
      

      <section className="artist-grid" aria-label="Artists">
        {artists.map((artist) => {
          const artistDisplayName = getArtistDisplayName(artist);
          return (
            <button className="artist-tile" key={artist.name} onClick={() => onSelect(artist)}>
              <img src={artist.imageUrl} alt={artistDisplayName} />
              <span>{artistDisplayName}</span>
            </button>
          );
        })}
      </section>
    </main>
  );
}

function Studio({ artist, activeAction, onActionChange, onBack, onHome, onAbout, onStudio, onContact }) {
  const active = actions.find((action) => action.id === activeAction);
  const activeCopy = active ? toolCopy[activeAction] : null;
  const artistDisplayName = getArtistDisplayName(artist);

  return (
    <main className="app-shell studio-shell">
      <Header onHome={onHome} onAbout={onAbout} onStudio={onStudio} onContact={onContact} />
      <header className="studio-header">
        <div>
          <h1>{artistDisplayName}</h1>
        </div>
      </header>

      <section className="studio-layout">
        <div className="workspace">
          <div className="action-tabs" role="tablist" aria-label="Studio actions">
            {actions.map((action) => (
              <button
                key={action.id}
                className={activeAction === action.id ? "active" : ""}
                onClick={() => onActionChange(action.id)}
                title={action.label}
                type="button"
              >
                <span>{action.label}</span>
              </button>
            ))}
          </div>

          <section className="tool-panel">
            {!activeAction && (
              <div className="empty-state">
                <h3 style={{ fontWeight: 400 }}>Choose an option to begin</h3>
                <h4 style={{ fontWeight: 400 }}>
                  Select a studio tool above to chat, generate an image, or redesign a picture.
                </h4>
              </div>
            )}

            {activeAction && (
              <>
                <h3 style={{ fontWeight: 400 }}>{activeCopy.title}</h3>
                <h4 style={{ fontWeight: 400 }}>
                  {activeCopy.description}
                </h4>

                {activeAction === "chat" && <ChatTool artist={artist} />}
                {activeAction === "paint" && <TextToImageTool artist={artist} />}
                {activeAction === "redesign" && <ImageToImageTool artist={artist} />}
              </>
            )}
          </section>
        </div>

        <aside className="artist-panel">
          <img src={artist.imageUrl} alt={artistDisplayName} />
          <button className="return-studio-button" onClick={onBack} type="button">
            Return to Studio
          </button>
        </aside>
      </section>
    </main>
  );
}

function ChatTool({ artist }) {
  const artistDisplayName = getArtistDisplayName(artist);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const messagesEndRef = useRef(null);

  useEffect(() => {
    setMessages([{ role: "assistant", content: `Hello, I am ${artistDisplayName}` }]);
    setQuestion("");
    setError("");
  }, [artist.name, artistDisplayName]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy]);

  async function submit(event) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || busy) return;

    setError("");
    setQuestion("");
    setMessages((current) => [...current, { role: "user", content: trimmed }]);
    setBusy(true);

    try {
      const history = messages.slice(-8);
      const response = await apiJson("/api/chat", {
        method: "POST",
        body: JSON.stringify({ artist: artist.name, question: trimmed, history })
      });
      setMessages((current) => [...current, { role: "assistant", content: response.Answer }]);
    } catch (err) {
      setError(getApiErrorMessage(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="chat-tool">
      <div className="message-list">
        {messages.length === 0 && (
          <div className="empty-state">
            <Bot size={28} />
            <p>Ask a question and {artistDisplayName} will answer using retrieved source context.</p>
          </div>
        )}
        {messages.map((message, index) => (
          <div className={`message ${message.role}`} key={`${message.role}-${index}`}>
            {message.content}
          </div>
        ))}
        {busy && <div className="message assistant">Thinking...</div>}
        <div ref={messagesEndRef} />
      </div>
      {error && <p className="error-text">{error}</p>}
      <form className="chat-input" onSubmit={submit}>
        <input
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="Enter your question for this artist"
        />
        <button type="submit" disabled={busy || !question.trim()} aria-label="Send question">
          <Send size={18} />
        </button>
      </form>
    </div>
  );
}

function TextToImageTool({ artist }) {
  const [description, setDescription] = useState("");
  const [imageUrl, setImageUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    setDescription("");
    setImageUrl("");
    setError("");
  }, [artist.name]);

  async function generate() {
    if (!description.trim()) {
      setError("Please write a description.");
      return;
    }

    setBusy(true);
    setError("");
    setImageUrl("");

    try {
      const response = await apiJson("/api/text-to-image", {
        method: "POST",
        body: JSON.stringify({ artist: artist.name, description })
      });
      setImageUrl(response.imageUrl);
    } catch (err) {
      setError(getApiErrorMessage(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="generator-tool">
      <label>
        Describe the scene
        <textarea
          value={description}
          onChange={(event) => setDescription(event.target.value)}
          rows={7}
        />
      </label>
      <button className="primary-button" onClick={generate} disabled={busy}>
        <Sparkles size={18} />
        {busy ? "Generating..." : "Generate Image"}
      </button>
      {error && <p className="error-text">{error}</p>}
      {imageUrl && (
        <figure className="result-image">
          <img src={imageUrl} alt="Generated artwork" />
          <figcaption>Generated Image</figcaption>
        </figure>
      )}
    </div>
  );
}

function ImageToImageTool({ artist }) {
  const [file, setFile] = useState(null);
  const [strength, setStrength] = useState(0.65);
  const [resultUrl, setResultUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const previewUrl = useMemo(() => {
    if (!file) return "";
    return URL.createObjectURL(file);
  }, [file]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  useEffect(() => {
    setFile(null);
    setStrength(0.65);
    setResultUrl("");
    setError("");
  }, [artist.name]);

  async function generate() {
    if (!file) {
      setError("Please upload an image first.");
      return;
    }

    const formData = new FormData();
    formData.append("artist", artist.name);
    formData.append("strength", String(strength));
    formData.append("image", file);

    setBusy(true);
    setError("");
    setResultUrl("");

    try {
      const response = await fetch("/api/image-to-image", {
        method: "POST",
        body: formData
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.detail || "Request failed.");
      }
      setResultUrl(payload.imageUrl);
    } catch (err) {
      setError(getApiErrorMessage(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="generator-tool">
      <label className="upload-box">
        <Upload size={22} />
        <span>{file ? file.name : "Upload a JPG or PNG image"}</span>
        <input
          type="file"
          accept="image/jpeg,image/png"
          onChange={(event) => setFile(event.target.files?.[0] || null)}
        />
      </label>

      <label>
        Style Strength: {strength.toFixed(2)}
        <input
          type="range"
          min="0.1"
          max="1"
          step="0.05"
          value={strength}
          onChange={(event) => setStrength(Number(event.target.value))}
        />
      </label>

      <button className="primary-button" onClick={generate} disabled={busy}>
        <Sparkles size={18} />
        {busy ? "Drawing..." : "Generate Image"}
      </button>
      {error && <p className="error-text">{error}</p>}

      <div className="image-results">
        {previewUrl && (
          <figure className="result-image">
            <img src={previewUrl} alt="Original upload" />
            <figcaption>Original Image</figcaption>
          </figure>
        )}
        {resultUrl && (
          <figure className="result-image">
            <img src={resultUrl} alt="Styled result" />
            <figcaption>Styled Result</figcaption>
          </figure>
        )}
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")).render(
  <>
    <App />
    <Analytics />
  </>
);
