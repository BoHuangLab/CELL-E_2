import useIntersectionObserver from '@react-hook/intersection-observer';
import { useRef } from 'react';

const LazyVideo = () => {
  const containerRef = useRef();
  const lockRef = useRef(false);
  const { isIntersecting } = useIntersectionObserver(containerRef);
  if (isIntersecting) {
    lockRef.current = true;
  }
  return (
    <div className="mb-4" ref={containerRef}>
      {lockRef.current && (
        <video
          className="sm:w-[800px]"
          autoPlay
          muted
          loop
          playsInline
          controls
          poster="/CELL-E_2/comparison_poster.jpg"
        >
          <source src="/CELL-E_2/comparison.webm" type="video/webm" />
        </video>
      )}
    </div>
  );
};

export default LazyVideo;
