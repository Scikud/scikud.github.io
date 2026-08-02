export const SITE = {
  title: 'ForceMultiplied',
  description: 'Notes on machine learning research, by Kudzo Ahegbebu.',
  author: 'Kudzo Ahegbebu',
};

export const formatDate = (date: Date) =>
  date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    timeZone: 'UTC',
  });

export const readingTime = (body: string) =>
  Math.max(1, Math.round(body.trim().split(/\s+/).length / 200));
