/** Star toggle for a page. Used by the TopBar (current page) and by each
 *  destination row in the search panel. Display-only preference. */
import { isFavourite, toggleFavourite, useFavourites, type Favourite } from './favourites';
import styles from './StarButton.module.css';

interface Props {
  target: Favourite | null;
  /** Slightly smaller inside search rows than in the bar. */
  compact?: boolean;
}

export default function StarButton({ target, compact }: Props) {
  useFavourites(); // re-render when the list changes anywhere in the app
  if (!target) return null;
  const on = isFavourite(target.to);
  const label = on ? `Remove ${target.label} from favourites` : `Add ${target.label} to favourites`;

  return (
    <button
      type="button"
      className={`${styles.star} ${on ? styles.on : ''} ${compact ? styles.compact : ''}`}
      title={label}
      aria-label={label}
      aria-pressed={on}
      onClick={(e) => {
        e.preventDefault();
        e.stopPropagation();
        toggleFavourite(target);
      }}
    >
      <svg viewBox="0 0 24 24" fill={on ? 'currentColor' : 'none'} stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round">
        <path d="m12 3.6 2.6 5.3 5.8.8-4.2 4.1 1 5.8-5.2-2.7-5.2 2.7 1-5.8L3.6 9.7l5.8-.8Z" />
      </svg>
    </button>
  );
}
