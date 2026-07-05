import { useEffect, useMemo, useRef, useState } from "react";

interface CatalogMovie {
  movieId: number;
  title: string;
  genres: string[];
  imdbUrl: string | null;
  tmdbUrl: string | null;
  posterUrl: string | null;
  overview: string | null;
  ratingCount: number;
}

interface RecEntry {
  movieId: number;
  score: number;
}

interface CatalogPayload {
  meta: { systems: string[]; top_k: number; n_query_movies: number; n_movies: number };
  movies: CatalogMovie[];
  queryMovieIds: number[];
  recommendations: Record<string, Record<string, RecEntry[]>>;
}

const RANDOM_PICKS_COUNT = 10;

// MovieLens stores titles with a trailing article, e.g. "Matrix, The (1999)",
// so a plain substring search for "The Matrix" never matches. Strip
// punctuation and drop leading/trailing articles from both the query and the
// title before comparing, and require every query word to appear somewhere
// in the title (order-independent) instead of one exact contiguous phrase.
const ARTICLES = new Set(["the", "a", "an"]);

function searchTokens(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[,():]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 0 && !ARTICLES.has(t));
}

function matchesQuery(title: string, query: string): boolean {
  const queryTokens = searchTokens(query);
  if (queryTokens.length === 0) return false;
  const titleWords = searchTokens(title);
  // Prefix match per word (not a free substring scan) — a query token like
  // "la" must start a word ("la", "land") rather than match inside any word
  // that happens to contain those letters ("Holland's", "Island", "Orlando").
  return queryTokens.every((token) => titleWords.some((word) => word.startsWith(token)));
}

// MovieLens stores the article at the end for sorting ("Matrix, The (1999)")
// — fine for search, but reads oddly everywhere it's displayed. Move it back
// to the front for display only ("The Matrix (1999)").
function displayTitle(title: string): string {
  const m = title.match(/^(.*), (The|A|An)((?:\s*\(.*\))*)$/i);
  if (!m) return title;
  const [, base, article, suffix] = m;
  return `${article} ${base}${suffix}`.trim();
}

function SearchIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="text-white/40"
    >
      <circle cx="11" cy="11" r="7" />
      <path d="m21 21-4.3-4.3" />
    </svg>
  );
}

const SYSTEM_LABELS: Record<string, string> = {
  content_cosine: "Content (cosine)",
  svd_collaborative: "Collaborative (SVD)",
};

const SYSTEM_HINTS: Record<string, string> = {
  content_cosine: "Similarity over the autoencoder embedding — best for genre-based discovery.",
  svd_collaborative: "SVD latent factors over the rating history — best real-behavior signal.",
};

function pickRandom(catalog: CatalogPayload, count: number): CatalogMovie[] {
  const byId = new Map(catalog.movies.map((m) => [m.movieId, m]));
  const shuffled = [...catalog.queryMovieIds].sort(() => Math.random() - 0.5);
  const picks: CatalogMovie[] = [];
  for (const id of shuffled) {
    const movie = byId.get(id);
    if (movie) picks.push(movie);
    if (picks.length >= count) break;
  }
  return picks;
}

function PosterCard({ movie, subtitle, onClick }: { movie: CatalogMovie; subtitle?: string; onClick?: () => void }) {
  return (
    <div
      onClick={onClick}
      className={`rounded-2xl border border-white/10 bg-black/30 backdrop-blur-xl overflow-hidden ${
        onClick ? "cursor-pointer hover:border-white/30 transition" : ""
      }`}
    >
      {movie.posterUrl ? (
        <img src={movie.posterUrl} alt={movie.title} className="w-full aspect-[2/3] object-cover" />
      ) : (
        <div className="w-full aspect-[2/3] flex items-center justify-center text-[11px] text-white/50 text-center px-2 bg-white/5">
          no poster
        </div>
      )}
      <div className="p-2">
        <div className="text-xs font-medium leading-snug line-clamp-2">{displayTitle(movie.title)}</div>
        <div className="text-[10px] text-white/50 mt-0.5 line-clamp-1">{movie.genres.join(" · ")}</div>
        {subtitle && <div className="text-[10px] text-sky-300/80 mt-1">{subtitle}</div>}
      </div>
    </div>
  );
}

export default function RecommenderView() {
  const [catalog, setCatalog] = useState<CatalogPayload | null>(null);
  const [randomPicks, setRandomPicks] = useState<CatalogMovie[]>([]);
  const [query, setQuery] = useState("");
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [system, setSystem] = useState<string>("content_cosine");
  const [focused, setFocused] = useState(false);
  const searchBoxRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function onOutsideClick(e: MouseEvent) {
      if (searchBoxRef.current && !searchBoxRef.current.contains(e.target as Node)) {
        setDropdownOpen(false);
      }
    }
    document.addEventListener("mousedown", onOutsideClick);
    return () => document.removeEventListener("mousedown", onOutsideClick);
  }, []);

  useEffect(() => {
    fetch("/recommender_catalog.json")
      .then((res) => res.json())
      .then((data: CatalogPayload) => {
        setCatalog(data);
        setRandomPicks(pickRandom(data, RANDOM_PICKS_COUNT));
      })
      .catch((err) => console.error("Failed to load recommender_catalog.json:", err));
  }, []);

  const movieById = useMemo(() => {
    if (!catalog) return new Map<number, CatalogMovie>();
    return new Map(catalog.movies.map((m) => [m.movieId, m]));
  }, [catalog]);

  const searchResults = useMemo(() => {
    if (!catalog || query.trim().length < 2) return [];
    const results: CatalogMovie[] = [];
    for (const id of catalog.queryMovieIds) {
      const movie = movieById.get(id);
      if (movie && matchesQuery(movie.title, query)) {
        results.push(movie);
        if (results.length >= 8) break;
      }
    }
    return results;
  }, [catalog, query, movieById]);

  const selectedMovie = selectedId != null ? movieById.get(selectedId) ?? null : null;
  const recs = useMemo(() => {
    if (!catalog || selectedId == null) return [];
    const entries = catalog.recommendations[system]?.[String(selectedId)] ?? [];
    return entries
      .map((e) => ({ ...e, movie: movieById.get(e.movieId) }))
      .filter((e): e is RecEntry & { movie: CatalogMovie } => Boolean(e.movie));
  }, [catalog, selectedId, system, movieById]);

  function selectMovie(movie: CatalogMovie) {
    setSelectedId(movie.movieId);
    setQuery(movie.title);
    setDropdownOpen(false);
  }

  function clearSelection() {
    setSelectedId(null);
    setQuery("");
    setDropdownOpen(false);
  }

  if (!catalog) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="h-10 w-10 rounded-full border-2 border-white/15 border-t-white/70 animate-spin" />
      </div>
    );
  }

  const searchActive = focused || query.length > 0;

  return (
    <div className="max-w-5xl mx-auto">
      <div ref={searchBoxRef} className="relative max-w-sm">
        <div
          className={`relative flex items-center gap-2.5 rounded-full px-4 py-2.5 border border-white/15 transition-all duration-300 ${
            searchActive ? "bg-black/70 backdrop-blur-xl shadow-xl" : "bg-transparent shadow-none"
          }`}
        >
          <SearchIcon />
          <input
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setSelectedId(null);
              setDropdownOpen(true);
            }}
            onFocus={() => {
              setFocused(true);
              setDropdownOpen(true);
            }}
            onBlur={() => setFocused(false)}
            placeholder="Search for a movie…"
            className="w-full bg-transparent text-sm text-white placeholder:text-white/30 outline-none"
          />
          {query.length > 0 && (
            <button
              onClick={clearSelection}
              aria-label="Clear search"
              className="text-white/40 hover:text-white transition text-sm shrink-0"
            >
              ✕
            </button>
          )}
        </div>
        {dropdownOpen && searchResults.length > 0 && (
          <div className="absolute top-full mt-2 w-full rounded-2xl border border-white/10 bg-black/70 backdrop-blur-xl shadow-xl overflow-hidden z-10">
            {searchResults.map((m) => (
              <button
                key={m.movieId}
                onClick={() => selectMovie(m)}
                className="flex items-center gap-2 w-full text-left px-3 py-2 text-sm bg-black/40 hover:bg-white/10 transition"
              >
                {m.posterUrl ? (
                  <img src={m.posterUrl} alt="" className="w-8 h-11 object-cover rounded shrink-0" />
                ) : (
                  <div className="w-8 h-11 rounded bg-white/5 shrink-0" />
                )}
                <span className="truncate">{displayTitle(m.title)}</span>
              </button>
            ))}
          </div>
        )}
      </div>

      {!selectedMovie && (
        <div className="mt-12">
          <div className="text-center text-[11px] uppercase tracking-wide text-white/50 mb-4">
            Or browse the catalog
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3">
            {randomPicks.map((m) => (
              <PosterCard key={m.movieId} movie={m} onClick={() => selectMovie(m)} />
            ))}
          </div>
        </div>
      )}

      {selectedMovie && (
        <div className="mt-10">
          <div className="flex flex-col sm:flex-row gap-4 rounded-2xl bg-black/30 backdrop-blur-xl p-4">
            {selectedMovie.posterUrl ? (
              <img src={selectedMovie.posterUrl} alt={selectedMovie.title} className="w-28 rounded-md object-cover shrink-0" />
            ) : (
              <div className="w-28 h-40 rounded-md bg-white/5 flex items-center justify-center text-[11px] text-white/50 text-center px-1 shrink-0">
                no poster
              </div>
            )}
            <div className="flex-1 min-w-0">
              <div className="text-lg font-semibold">{displayTitle(selectedMovie.title)}</div>
              <div className="text-xs text-white/50 mt-1">{selectedMovie.genres.join(" · ")}</div>
              <div className="text-xs text-white/50 mt-1">{selectedMovie.ratingCount.toLocaleString()} ratings</div>
              {selectedMovie.overview && (
                <p className="text-xs text-white/60 mt-2 leading-relaxed line-clamp-4">{selectedMovie.overview}</p>
              )}
              <div className="flex gap-3 mt-2 text-xs">
                {selectedMovie.imdbUrl && (
                  <a href={selectedMovie.imdbUrl} target="_blank" rel="noreferrer" className="text-amber-300 hover:underline">
                    IMDb
                  </a>
                )}
                {selectedMovie.tmdbUrl && (
                  <a href={selectedMovie.tmdbUrl} target="_blank" rel="noreferrer" className="text-sky-300 hover:underline">
                    TMDb
                  </a>
                )}
              </div>
            </div>
          </div>

          <div className="flex items-center justify-center gap-2 mt-6">
            {catalog.meta.systems.map((s) => (
              <button
                key={s}
                onClick={() => setSystem(s)}
                className={`px-4 py-1.5 rounded-full text-sm font-medium border transition ${
                  system === s
                    ? "bg-white/10 border-white/30 text-white"
                    : "border-white/10 text-white/60 hover:text-white"
                }`}
              >
                {SYSTEM_LABELS[s] ?? s}
              </button>
            ))}
          </div>
          <div className="text-center text-[11px] text-white/40 mt-2 max-w-md mx-auto">
            {SYSTEM_HINTS[system]}
          </div>

          <div className="text-center text-[11px] uppercase tracking-wide text-white/50 mt-8 mb-4">
            Recommended for you
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3">
            {recs.map((r) => (
              <PosterCard key={r.movieId} movie={r.movie} subtitle={`score ${r.score.toFixed(3)}`} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
