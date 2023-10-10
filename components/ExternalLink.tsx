type ButtonProps = {
  children: React.ReactNode;
  href: string;
};

const ExternalLink = ({ children, ...props }: ButtonProps) => {
  return (
    <a
      className="text-indigo-500 hover:text-zinc-900 decoration-2 hover:underline font-bold"
      target="_blank"
      rel="noopener noreferrer"
      {...props}
    >
      {children}
    </a>
  );
};

export default ExternalLink;
