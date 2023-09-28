import { useState } from 'react';

type DemoProps = {
  id: string;
  src: string;
  height: string;
};

function Demo({ id, src, height }: DemoProps) {
  const [showDemo, setShowDemo] = useState(false);

  return (
    <div className="flex justify-center mt-12 mb-16 px-2 xl:px-32">
      {showDemo ? (
        <iframe
          id={id}
          className="rounded-lg w-full xl:w-4/5"
          src={src}
          frameBorder="0"
          scrolling="no"
          height={height}
        ></iframe>
      ) : (
        <button
          style={{
            height: `${Number(height) / 3}px`
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
