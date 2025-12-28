// ============================================
// SEND LAUNCH NOTIFICATION - Supabase Edge Function
// Sends emails to waitlist subscribers when project goes live
// ============================================
//
// Deploy: supabase functions deploy send-launch-notification
//
// Environment variables required:
// - RESEND_API_KEY: API key from resend.com
// - SUPABASE_URL: Auto-provided by Supabase
// - SUPABASE_SERVICE_ROLE_KEY: Auto-provided by Supabase
// ============================================

import { serve } from 'https://deno.land/std@0.168.0/http/server.ts';
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

const RESEND_API_KEY = Deno.env.get('RESEND_API_KEY');
const SUPABASE_URL = Deno.env.get('SUPABASE_URL');
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY');

interface LaunchPayload {
  project_id: string;
  project_name: string;
  project_slug: string;
}

interface WaitlistSubscriber {
  id: string;
  email: string;
}

serve(async (req: Request) => {
  // Only allow POST requests
  if (req.method !== 'POST') {
    return new Response('Method not allowed', { status: 405 });
  }

  try {
    const payload: LaunchPayload = await req.json();
    const { project_id, project_name, project_slug } = payload;

    if (!project_id || !project_name) {
      return new Response(
        JSON.stringify({ error: 'Missing project_id or project_name' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      );
    }

    // Initialize Supabase client with service role
    const supabase = createClient(
      SUPABASE_URL!,
      SUPABASE_SERVICE_ROLE_KEY!
    );

    // Get all subscribers who haven't been notified
    const { data: subscribers, error: fetchError } = await supabase
      .from('waitlist')
      .select('id, email')
      .eq('project_id', project_id)
      .eq('notified', false);

    if (fetchError) {
      throw new Error(`Failed to fetch subscribers: ${fetchError.message}`);
    }

    if (!subscribers || subscribers.length === 0) {
      return new Response(
        JSON.stringify({ message: 'No subscribers to notify', count: 0 }),
        { status: 200, headers: { 'Content-Type': 'application/json' } }
      );
    }

    console.log(`Sending notifications to ${subscribers.length} subscribers for ${project_name}`);

    // Send emails via Resend
    const emailPromises = subscribers.map((sub: WaitlistSubscriber) =>
      sendEmail(sub.email, project_name, project_slug)
    );

    const results = await Promise.allSettled(emailPromises);

    // Count successes and failures
    const successful = results.filter(r => r.status === 'fulfilled').length;
    const failed = results.filter(r => r.status === 'rejected').length;

    // Mark successful subscribers as notified
    if (successful > 0) {
      const successfulEmails = subscribers
        .filter((_, index) => results[index].status === 'fulfilled')
        .map(sub => sub.id);

      await supabase
        .from('waitlist')
        .update({ notified: true, notified_at: new Date().toISOString() })
        .in('id', successfulEmails);
    }

    return new Response(
      JSON.stringify({
        message: `Notifications sent for ${project_name}`,
        total: subscribers.length,
        successful,
        failed
      }),
      { status: 200, headers: { 'Content-Type': 'application/json' } }
    );

  } catch (error) {
    console.error('Error sending notifications:', error);
    return new Response(
      JSON.stringify({ error: error.message }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
  }
});

// ============================================
// Send individual email via Resend
// ============================================
async function sendEmail(
  to: string,
  projectName: string,
  projectSlug: string
): Promise<void> {
  const projectUrl = `https://castronix.dev/${projectSlug}`;

  const response = await fetch('https://api.resend.com/emails', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${RESEND_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      from: 'Castronix <launch@castronix.dev>',
      to: [to],
      subject: `${projectName} is now LIVE!`,
      html: generateEmailHtml(projectName, projectSlug, projectUrl),
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Failed to send email to ${to}: ${error}`);
  }
}

// ============================================
// Generate email HTML template
// ============================================
function generateEmailHtml(
  projectName: string,
  projectSlug: string,
  projectUrl: string
): string {
  return `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; background-color: #0A0A0F; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #0A0A0F; padding: 40px 20px;">
    <tr>
      <td align="center">
        <table width="600" cellpadding="0" cellspacing="0" style="background-color: #0F1117; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);">
          <!-- Header -->
          <tr>
            <td style="padding: 40px 40px 20px; text-align: center;">
              <div style="width: 48px; height: 48px; background: linear-gradient(135deg, #22D3EE 0%, #0EA5E9 100%); border-radius: 10px; display: inline-block; line-height: 48px; color: #0A0A0F; font-weight: bold; font-size: 20px;">C</div>
              <h1 style="color: #FAFAFA; font-size: 28px; font-weight: 600; margin: 20px 0 0; letter-spacing: -0.02em;">The wait is over.</h1>
            </td>
          </tr>

          <!-- Content -->
          <tr>
            <td style="padding: 20px 40px 30px;">
              <p style="color: #A1A1A1; font-size: 16px; line-height: 1.6; margin: 0 0 24px;">
                <strong style="color: #22D3EE;">${projectName}</strong> is now live and ready for you to explore.
              </p>
              <p style="color: #A1A1A1; font-size: 16px; line-height: 1.6; margin: 0 0 32px;">
                Thank you for your interest and patience. We've built something we think you'll love.
              </p>

              <!-- CTA Button -->
              <table cellpadding="0" cellspacing="0" style="margin: 0 auto;">
                <tr>
                  <td style="background: #FAFAFA; border-radius: 8px;">
                    <a href="${projectUrl}" style="display: inline-block; padding: 14px 32px; color: #0A0A0F; text-decoration: none; font-weight: 600; font-size: 14px;">
                      Check out ${projectName} &rarr;
                    </a>
                  </td>
                </tr>
              </table>
            </td>
          </tr>

          <!-- Footer -->
          <tr>
            <td style="padding: 24px 40px; border-top: 1px solid rgba(255,255,255,0.06);">
              <p style="color: #525252; font-size: 12px; margin: 0; text-align: center;">
                You received this because you signed up for ${projectName} updates at castronix.dev
              </p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>
  `.trim();
}
