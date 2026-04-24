import {
	WorkflowEntrypoint,
	type WorkflowEvent,
	type WorkflowStep
} from "cloudflare:workers";
import { createWorkersAI } from "workers-ai-provider";
import { generateText } from "ai";

type ResearchParams = {
	question: string;
};

export class ResearchWorkflow extends WorkflowEntrypoint<Env, ResearchParams> {
	async run(event: WorkflowEvent<ResearchParams>, step: WorkflowStep) {
		const { question } = event.payload;
		const workersai = createWorkersAI({ binding: this.env.AI });
		const model = workersai("@cf/meta/llama-3.3-70b-instruct-fp8-fast");

		
		// Step 1: Break the question into 3 focused sub-questions
		const subQuestions = await step.do("decompose-question", async () => {
			const { text } = await generateText({
				model,
				prompt: `Break this research question into exactly 3 specific sub-questions that together would fully answer it.
Question: "${question}"
Return ONLY a JSON array of 3 strings, no explanation. Example: ["sub-q 1", "sub-q 2", "sub-q 3"]`
			});
			try {
				const match = text.match(/\[[\s\S]*\]/);
				return JSON.parse(match ? match[0] : "[]") as string[];
			} catch {
				return [question];
			}
		});

		// Step 2: Answer each sub-question in parallel
		const answers = await step.do("research-sub-questions", async () => {
			const results = await Promise.all(
				subQuestions.map(async (subQ: string) => {
					const { text } = await generateText({
						model,
						prompt: `Answer this question concisely in 2-3 sentences: "${subQ}"`
					});
					return { question: subQ, answer: text };
				})
			);
			return results;
		});

		// Step 3: Synthesize all answers into a final comprehensive response
		const synthesis = await step.do("synthesize-answer", async () => {
			const context = answers
				.map(
					(a: { question: string; answer: string }) =>
						`Q: ${a.question}\nA: ${a.answer}`
				)
				.join("\n\n");

			const { text } = await generateText({
				model,
				prompt: `Using these research findings, write a clear and comprehensive answer to the original question.

Original question: "${question}"

Research findings:
${context}

Write a well-structured response that synthesizes all the findings.`
			});
			return text;
		});

		return {
			originalQuestion: question,
			subQuestions,
			answers,
			synthesis
		};
	}
}
