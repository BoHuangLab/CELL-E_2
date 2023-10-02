import { useState } from 'react';
import IframeResizer from 'iframe-resizer-react';

type DemoProps = {
  id: string;
  src: string;
  height?: string;
  className?: string;
};

function Demo({ id, src, height, className = '' }: DemoProps) {
  const [showDemo, setShowDemo] = useState(false);

  return (
    <div className="flex justify-center mt-12 mb-16 px-2 xl:px-8 2xl:px-32">
      {showDemo ? (
        <IframeResizer
          id={id}
          className={`rounded-lg w-full xl:w-4/5 ${className}`}
          src={src}
          frameBorder="0"
          scrolling={false} // == `scrolling="no"`
          style={{
            minWidth: '100%',
            minHeight: 1000
          }}
        />
      ) : (
        <button
          style={{
            height: height ? `${Number(height) / 3}px` : 400
          }}
          className={`
            flex justify-center items-center
            w-full xl:w-4/5
            bg-gray-100 text-indigo-500 hover:bg-gray-200
            font-semibold
            rounded-lg
            cursor-pointer
            transition
            border-[1px] border-dashed border-zinc-300 hover:border-gray-200
          `}
          onClick={() => setShowDemo(true)}
        >
          Click to load demo
        </button>
      )}
    </div>
  );
}

export default Demo;
