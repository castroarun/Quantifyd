import type { ReactNode } from 'react';
import styles from './DataTable.module.css';

export interface Column<T> {
  key: string;
  header: string;
  width?: string;        // grid-template-columns fragment, e.g. '1.2fr', '90px'
  align?: 'left' | 'right' | 'center';
  render: (row: T, idx: number) => ReactNode;
}

interface Props<T> {
  columns: Column<T>[];
  rows: T[];
  emptyText?: string;
  rowKey?: (row: T, idx: number) => string | number;
}

export default function DataTable<T>({
  columns,
  rows,
  emptyText = 'No data',
  rowKey,
}: Props<T>) {
  const template = columns.map((c) => c.width ?? 'minmax(0, 1fr)').join(' ');

  return (
    <div className={styles.table}>
      <div
        className={styles.head}
        style={{ display: 'grid', gridTemplateColumns: template }}
      >
        {columns.map((c) => (
          <div
            key={c.key}
            className={styles.headCell}
            style={{ textAlign: c.align ?? 'left' }}
          >
            {c.header}
          </div>
        ))}
      </div>

      {rows.length === 0 ? (
        <div className={styles.empty}>{emptyText}</div>
      ) : (
        rows.map((row, idx) => (
          <div
            key={rowKey ? rowKey(row, idx) : idx}
            className={styles.row}
            style={{ display: 'grid', gridTemplateColumns: template }}
          >
            {columns.map((c) => (
              <div
                key={c.key}
                className={styles.cell}
                style={{ textAlign: c.align ?? 'left' }}
              >
                {c.render(row, idx)}
              </div>
            ))}
          </div>
        ))
      )}
    </div>
  );
}
